"""
混合检索编排器（对齐《项目文档》5.4 全流程）

流程：Query Rewrite / HyDE（多路查询）
  → 双路召回：ES/BM25 关键词 + Milvus 向量（先权限过滤再召回）
  → RRF 融合（score = Σ 1/(k+rank)，k=60）
  → BGE-Reranker 重排 → Top-K 生成上下文
"""

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

    # ---- 召回 ----
    def retrieve(self, queries: List[str], top_k: Optional[int] = None,
                 tenant_id: str = "mihc", allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        多查询（原始问题 + 改写 + HyDE）双路召回 + RRF + 重排。

        Args:
            queries: 查询列表（至少包含原始问题；可追加改写查询与假设文档）
            top_k: 最终返回条数（默认取配置）
        Returns:
            重排后的片段列表（含 chunk_id/text/title/source/score/rerank_score）
        """
        top_k = top_k or self.config.reranker.top_k
        fused: Dict[str, Dict[str, Any]] = {}

        for query in queries:
            for q_text in [query] if isinstance(query, str) else [query]:
                # 双路召回（权限过滤先于向量召回，见文档 5.7）
                # 向量路故障只降级，不阻断关键词路（云端 Milvus 不可用时仍可回答）
                try:
                    dense_hits = self.milvus.search(
                        self.embedder.embed_query(q_text),
                        top_k=self.retrieval_cfg.top_k,
                        tenant_id=tenant_id,
                        allowed_doc_ids=allowed_doc_ids,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Dense retrieval degraded: %s", exc)
                    dense_hits = []
                sparse_hits = self.keyword.search(
                    q_text,
                    top_k=self.retrieval_cfg.keyword_top_k,
                    tenant_id=tenant_id,
                    allowed_doc_ids=allowed_doc_ids,
                )
                # RRF 融合
                self._rrf_fuse(fused, dense_hits, sparse_hits)

        candidates = list(fused.values())
        if not candidates:
            logger.info("Retrieval: no candidates for queries=%s", queries)
            return []

        # BGE-Reranker 重排（用原始问题重排）
        original_query = queries[0] if queries else ""
        reranked = self.reranker.rerank(original_query, candidates, top_k=top_k)
        logger.info("Retrieval: %d candidates -> %d after rerank", len(candidates), len(reranked))
        return reranked

    @staticmethod
    def _rrf_fuse(fused: Dict[str, Dict[str, Any]], dense: List[Dict[str, Any]],
                  sparse: List[Dict[str, Any]], k: int = 60) -> None:
        """RRF 融合：score = Σ 1/(k + rank)。同一 chunk 在关键词与语义结果中都靠前则得分更高。"""
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
