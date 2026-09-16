"""
从 HuggingFace 数据集提取 mIHC 知识点并写入 RAG 知识库。
流程：hf_hub_download 下载数据集 JSON → 关键词过滤 → 保存 markdown
     → 入库管线（解析→切分→嵌入→Milvus+BM25+PG）。

用法：
  python scripts/ingest_hf_knowledge.py --max-docs 60
  python scripts/ingest_hf_knowledge.py --datasets medalpaca/medical_meadow_wikidoc \
      --keywords "mIHC,multiplex immunofluorescence,tumor microenvironment" --max-docs 30
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASETS = [
    "medalpaca/medical_meadow_wikidoc",
    "medalpaca/medical_meadow_medical_flashcards",
]

DEFAULT_KEYWORDS = [
    "mIHC",
    "multiplex immunofluorescence",
    "tumor microenvironment",
    "tumor immune microenvironment",
    "immunofluorescence",
    "PD-L1",
    "CD8",
    "FOXP3",
    "immune infiltration",
    "spatial biology",
]


def download_dataset_files(dataset: str) -> list:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(dataset, repo_type="dataset")
             if f.endswith((".json", ".jsonl"))]
    paths = []
    for f in files:
        try:
            paths.append(hf_hub_download(dataset, f, repo_type="dataset"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("下载 %s 失败: %s", f, exc)
    return paths


def load_records(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        logger.warning("%s 不是列表结构，跳过", path)
        return
    yield from data


def pick_text(record) -> str:
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""
    for key in ("output", "text", "content", "response", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    candidates = [v for v in record.values() if isinstance(v, str) and v.strip()]
    return max(candidates, key=len) if candidates else ""


def main():
    parser = argparse.ArgumentParser(description="HuggingFace mIHC 知识入库")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="逗号分隔的 HF 数据集名（JSON/JSONL 结构）")
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS),
                        help="逗号分隔的过滤关键词")
    parser.add_argument("--max-docs", type=int, default=60, help="入库文档数上限")
    args = parser.parse_args()

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    out_dir = Path("./data/hf_docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    seen = set()
    for dataset in datasets:
        if len(selected) >= args.max_docs:
            break
        paths = download_dataset_files(dataset)
        for path in paths:
            for record in load_records(path):
                text = pick_text(record)
                if len(text) < 200 or text in seen:
                    continue
                lowered = text.lower()
                if not any(kw in lowered for kw in keywords):
                    continue
                seen.add(text)
                selected.append({"dataset": dataset, "text": text})
                if len(selected) >= args.max_docs:
                    break
            if len(selected) >= args.max_docs:
                break
        logger.info("数据集 %s 后累计命中 %d 条", dataset, len(selected))

    if not selected:
        print("未检索到匹配记录：请调整 --keywords 或 --datasets")
        return

    for i, item in enumerate(selected):
        path = out_dir / f"hf_{i:04d}.md"
        path.write_text(item["text"], encoding="utf-8")

    from config import Config
    from services.service import MIHCPlatform

    platform = MIHCPlatform(Config())
    ok = failed = 0
    for i, item in enumerate(selected):
        path = out_dir / f"hf_{i:04d}.md"
        try:
            result = platform.ingest_document(
                str(path),
                source=f"huggingface:{item['dataset']}",
                permission="research_team",
            )
            ok += 1
            logger.info("[%d/%d] 入库 %s: %s 个片段", i + 1, len(selected), result.file_name, result.chunks)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("入库失败 %s: %s", path.name, exc)

    print(f"完成：成功 {ok} 篇 / 失败 {failed} 篇（共 {len(selected)} 篇）")


if __name__ == "__main__":
    main()
