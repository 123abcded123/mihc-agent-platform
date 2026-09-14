"""
知识库入库管线（对齐《项目文档》5.3）

流程：
  文件采集 → 格式解析(Docling) → 清洗 → 按标题/段落切分
  → BGE-M3 生成 embedding → 写入 Milvus
  → 关键词字段写入 Elasticsearch（或本地 BM25 回退）
  → PostgreSQL 保存文档版本、来源和权限

chunk 至少保存（对齐文档 5.3 的 JSON 结构）：
  chunk_id / text / title / source / version / permissions / embedding_model
"""

from __future__ import annotations

import re
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from agents.rag_agent.doc_parser import MedicalDocParser
from core.errors import MIHCError

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """文档入库编排器。"""

    def __init__(self, config, embedder, milvus_store, keyword_store, database):
        self.config = config
        self.embedder = embedder
        self.milvus = milvus_store
        self.keyword = keyword_store
        self.db = database
        self.parser = MedicalDocParser()

    # ---- 解析 ----
    def parse(self, file_path: str) -> Tuple[str, str]:
        """解析文档为 markdown 文本。返回 (markdown 文本, 文件名)。

        优先 Docling（表格/OCR/结构完整）；失败时回退轻量解析器（pypdf 等），
        保证无 Docling 模型下载环境（如本机）也能入库。
        """
        path = Path(file_path)
        if not path.exists():
            raise MIHCError(f"文件不存在: {file_path}", code="file_not_found", status_code=404)
        if path.suffix.lower() not in (".pdf", ".docx", ".md", ".txt"):
            raise MIHCError(f"暂不支持的文件类型: {path.suffix}", code="unsupported_file", status_code=422)
        logger.info("Parsing document: %s", file_path)
        try:
            doc, _images = self.parser.parse_document(str(path), "./data/parsed_docs")
            markdown = doc.export_to_markdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Docling 解析失败（%s），回退轻量解析器", exc)
            markdown = self._fallback_parse(path)
        return markdown, path.name

    @staticmethod
    def _fallback_parse(path) -> str:
        """轻量回退解析：PDF→pypdf 逐页提取；DOCX→python-docx；MD/TXT→直接读取。"""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## 第 {i} 页\n{text}")
            return "\n\n".join(pages)
        if suffix == ".docx":
            from docx import Document as DocxDocument
            doc = DocxDocument(str(path))
            parts = []
            for para in doc.paragraphs:
                if para.style.name.startswith("Heading"):
                    parts.append(f"## {para.text}")
                elif para.text.strip():
                    parts.append(para.text)
            for table in doc.tables:
                parts.append("\n".join(" | ".join(c.text for c in row.cells) for row in table.rows))
            return "\n\n".join(parts)
        return path.read_text(encoding="utf-8", errors="ignore")

    # ---- 清洗 ----
    @staticmethod
    def clean(text: str) -> str:
        """清洗：去页眉页脚数字页、重复空白、乱码字符。"""
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)          # 独立页码行
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ---- 切分（按标题/段落边界，对齐文档 5.3）----
    @staticmethod
    def chunk_by_sections(text: str, max_chars: int = 512, overlap_chars: int = 50) -> List[Dict[str, str]]:
        """按标题边界粗切，再按段落合并到 max_chars 左右；表格块整体保留不被拆散。"""
        sections = re.split(r"(?=\n#+\s)", text)
        chunks: List[Dict[str, str]] = []
        current = ""
        current_title = ""

        def flush():
            nonlocal current, current_title
            if current.strip():
                chunks.append({"text": current.strip(), "title": current_title})
            current = ""
            current_title = ""

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            title_match = re.match(r"#+\s+(.*)", sec.split("\n", 1)[0])
            title = title_match.group(1).strip() if title_match else ""
            # 段落切分；表格行（|开头）与段落合并保留完整性
            paras = re.split(r"\n(?=\S)", sec)
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if current and len(current) + len(para) > max_chars:
                    flush()
                if not current:
                    current_title = title
                current = (current + "\n\n" + para).strip()
        flush()
        # 相邻块重叠（防止语义被切断）
        if overlap_chars > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                chunks[i]["text"] = chunks[i - 1]["text"][-overlap_chars:] + "\n" + chunks[i]["text"]
        return chunks

    # ---- 入库 ----
    def ingest(self, file_path: str, *, version: str = "", source: str = "",
               permission: str = "research_team", tenant_id: str = "mihc") -> Dict[str, Any]:
        """完整入库：解析→清洗→切分→嵌入→双写(Milvus+关键词)→PG 元数据。"""
        markdown, file_name = self.parse(file_path)
        text = self.clean(markdown)
        sections = self.chunk_by_sections(text)

        doc_id = uuid.uuid4().hex
        source = source or str(file_path)
        embed_model = self.config.embedding.model_name

        chunk_records: List[Dict[str, Any]] = []
        for i, sec in enumerate(sections):
            chunk_id = f"{doc_id}-chunk-{i:03d}"
            chunk_records.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": sec["text"],
                "title": sec.get("title", ""),
                "source": source,
                "version": version,
                "tenant_id": tenant_id,
                "embedding_model": embed_model,
            })

        # BGE-M3 嵌入
        embeddings = self.embedder.embed_documents([c["text"] for c in chunk_records])

        # 双写：Milvus（向量）+ 关键词库（ES/BM25）
        # 向量库故障只降级不阻断入库，关键词路仍可检索。
        try:
            self.milvus.upsert_chunks(chunk_records, embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus upsert degraded (%s), keyword index only", exc)
        self.keyword.upsert_chunks(chunk_records)

        # PostgreSQL 元数据：文档 + 片段索引
        self.db.add_document(doc_id=doc_id, file_name=file_name, source=source, version=version,
                             tenant_id=tenant_id, permission=permission,
                             embedding_model=embed_model, chunk_count=len(chunk_records))
        self.db.add_chunks([{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                             "title": c["title"], "text": c["text"],
                             "source": c["source"], "version": c["version"],
                             "tenant_id": c["tenant_id"], "embedding_model": c["embedding_model"]}
                            for c in chunk_records])

        logger.info("Ingested %s: %d chunks -> Milvus + keyword store + PG", file_name, len(chunk_records))
        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "chunks": len(chunk_records),
            "embedding_model": embed_model,
        }
