
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from core.errors import RetrievalError
from rag.retriever.milvus_store import MilvusStore
from rag.retriever.es_store import KeywordStore
from rag.retriever.reranker import create_reranker

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, config, embedder=None):
        self.config = config
        self.retrieval_cfg = config.retrieval
        self.embedder = embedder
        self.milvus = MilvusStore(config)
        self.keyword = KeywordStore(config)
        self.reranker = create_reranker(config)

    def retrieve(self, queries: List[str], top_k: Optional[int] = None,
                 tenant_id: str = "mihc", allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.reranker.top_k
        fused: Dict[str, Dict[str, Any]] = {}

        for query in queries:
            for q_text in [query] if isinstance(query, str) else [query]:
                try:
                    dense_hits = self.milvus.search(
                        self.embedder.embed_query(q_text),
                        top_k=self.retrieval_cfg.top_k,
                        tenant_id=tenant_id,
                        allowed_doc_ids=allowed_doc_ids,
                    )
                except Exception as exc:
                    logger.warning("Dense retrieval degraded: %s", exc)
                    dense_hits = []
                sparse_hits = self.keyword.search(
                    q_text,
                    top_k=self.retrieval_cfg.keyword_top_k,
                    tenant_id=tenant_id,
                    allowed_doc_ids=allowed_doc_ids,
                )
                self._rrf_fuse(fused, dense_hits, sparse_hits)

        candidates = list(fused.values())
        if not candidates:
            logger.info("Retrieval: no candidates for queries=%s", queries)
            return []

        original_query = queries[0] if queries else ""
        reranked = self.reranker.rerank(original_query, candidates, top_k=top_k)
        logger.info("Retrieval: %d candidates -> %d after rerank", len(candidates), len(reranked))
        return reranked

    @staticmethod
    def _rrf_fuse(fused: Dict[str, Dict[str, Any]], dense: List[Dict[str, Any]],
                  sparse: List[Dict[str, Any]], k: int = 60) -> None:
        for rank, hit in enumerate(dense, start=1):
            cid = hit.get("chunk_id", "")
            if not cid:
                continue
            item = fused.setdefault(cid, dict(hit))
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (k + rank)
        for rank, hit in enumerate(sparse, start=1):
            cid = hit.get("chunk_id", "")
            if not cid:
                continue
            item = fused.setdefault(cid, dict(hit))
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (k + rank)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "milvus_chunks": self.milvus.count(),
        }
