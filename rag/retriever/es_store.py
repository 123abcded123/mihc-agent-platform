"""
关键词检索存储（Elasticsearch / 本地 BM25 回退）

对齐《项目文档》5.4 第二步"双路召回"中的关键词路：
- 精确匹配药物名、基因名、检测指标、年份等；
- ES_URL 配置时走真实 Elasticsearch；
- 未配置时回退本地 BM25（rank-bm25 + JSON 持久化索引），接口一致。

索引单元与 Milvus chunk 一一对应（chunk_id 相同），便于 RRF 融合。
"""

from __future__ import annotations

import os
import json
import logging
import re
import threading
from typing import List, Dict, Any, Optional

from core.errors import RetrievalError

logger = logging.getLogger(__name__)

_TOKENIZE = re.compile(r"[\w\u4e00-\u9fff]+")


class _BaseKeywordStore:
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5, tenant_id: str = "mihc",
               allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError


class LocalBM25Store(_BaseKeywordStore):
    """本地 BM25 回退实现：rank-bm25 打分 + JSON 文件持久化。"""

    def __init__(self, index_path: str):
        self.index_path = index_path
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._bm25 = None
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
                self._rebuild()
                logger.info("BM25 local index loaded: %d chunks", len(self._records))
            except json.JSONDecodeError:
                logger.warning("BM25 index file corrupted, start empty")
                self._records = []

    def _rebuild(self):
        from rank_bm25 import BM25Okapi
        corpus = [self._tokenize(r["text"]) for r in self._records]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _persist(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in _TOKENIZE.findall(text or "")]

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        with self._lock:
            existing_ids = {r["chunk_id"] for r in self._records}
            for c in chunks:
                if c["chunk_id"] in existing_ids:
                    for i, r in enumerate(self._records):
                        if r["chunk_id"] == c["chunk_id"]:
                            self._records[i] = c
                            break
                else:
                    self._records.append(c)
            self._rebuild()
            self._persist()
        return len(chunks)

    def search(self, query: str, top_k: int = 5, tenant_id: str = "mihc",
               allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(self._tokenize(query))
            ranked = sorted(range(len(self._records)), key=lambda i: scores[i], reverse=True)
            out = []
            for idx in ranked:
                rec = self._records[idx]
                # 权限过滤：先于召回生效
                if rec.get("tenant_id", "mihc") != tenant_id:
                    continue
                if allowed_doc_ids and rec.get("doc_id") not in allowed_doc_ids:
                    continue
                out.append({
                    "chunk_id": rec["chunk_id"],
                    "text": rec.get("text", ""),
                    "title": rec.get("title", ""),
                    "source": rec.get("source", ""),
                    "version": rec.get("version", ""),
                    "doc_id": rec.get("doc_id", ""),
                    "score": float(scores[idx]),
                })
                if len(out) >= top_k:
                    break
            return out

    def delete_by_doc(self, doc_id: str) -> None:
        with self._lock:
            self._records = [r for r in self._records if r.get("doc_id") != doc_id]
            self._rebuild()
            self._persist()


class ElasticsearchStore(_BaseKeywordStore):
    """真实 Elasticsearch 后端（ES_URL 配置时启用）。"""

    def __init__(self, url: str, index: str):
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch(url)
        self.index = index
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "title": {"type": "text"},
                    "source": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                }
            })
            logger.info("ES index created: %s", self.index)

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        from elasticsearch.helpers import bulk
        actions = [
            {"_index": self.index, "_id": c["chunk_id"],
             "_source": {k: c.get(k, "") for k in
                         ("chunk_id", "text", "title", "source", "version", "tenant_id", "doc_id")}}
            for c in chunks
        ]
        bulk(self.client, actions)
        return len(chunks)

    def search(self, query: str, top_k: int = 5, tenant_id: str = "mihc",
               allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        must = [
            {"multi_match": {"query": query, "fields": ["text^2", "title^3"]}},
            {"term": {"tenant_id": tenant_id}},
        ]
        if allowed_doc_ids:
            must.append({"terms": {"doc_id": allowed_doc_ids}})
        try:
            resp = self.client.search(index=self.index, query={"bool": {"must": must}}, size=top_k)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Elasticsearch 检索失败: {exc}")
        return [
            {"chunk_id": h["_id"], "text": h["_source"].get("text", ""),
             "title": h["_source"].get("title", ""), "source": h["_source"].get("source", ""),
             "version": h["_source"].get("version", ""), "doc_id": h["_source"].get("doc_id", ""),
             "score": float(h["_score"])}
            for h in resp["hits"]["hits"]
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        self.client.delete_by_query(index=self.index, query={"term": {"doc_id": doc_id}})


class KeywordStore:
    """关键词检索门面：按配置选择 ES 或本地 BM25。"""

    def __init__(self, config):
        self.backend_name = "elasticsearch" if config.elastic.url else "bm25_local"
        if config.elastic.url:
            self.backend = ElasticsearchStore(config.elastic.url, config.elastic.index)
        else:
            self.backend = LocalBM25Store(config.elastic.local_index_path)
        logger.info("Keyword store backend: %s", self.backend_name)

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        return self.backend.upsert_chunks(chunks)

    def search(self, query: str, top_k: int = 5, tenant_id: str = "mihc",
               allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.backend.search(query, top_k, tenant_id, allowed_doc_ids)

    def delete_by_doc(self, doc_id: str) -> None:
        self.backend.delete_by_doc(doc_id)
