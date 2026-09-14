"""
BGE-Reranker 重排模块（对齐《项目文档》5.4 第四步）

BGE-Reranker 接收"问题 + 候选片段"，直接判断相关性，选出 Top-K 作为生成上下文。
Reranker 不能创造证据，只能重新排列已有候选。
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from core.errors import RetrievalError

logger = logging.getLogger(__name__)


class BGEReranker:
    """BGE-Reranker（优先本地预下载模型，回退 HuggingFace 运行时下载）。"""

    LOCAL_PATH = "./models/bge-reranker-base"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        import os
        if os.path.isdir(self.LOCAL_PATH):
            model_name = self.LOCAL_PATH
        logger.info("Loading reranker model %s on %s ...", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """对候选片段重排，返回 top_k 条（附 rerank_score 与 normalized score）。"""
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"BGE-Reranker 重排失败: {exc}")

        scored = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class ScoreReranker:
    """无重排模型时的降级实现：按 RRF 融合分排序，不引入模型评分。"""

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        scored = []
        for cand in candidates:
            item = dict(cand)
            item["rerank_score"] = float(cand.get("rrf_score", cand.get("score", 0.0)))
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


def create_reranker(config) -> object:
    """按配置与模型可用性选择重排器：BGE-Reranker 或得分降级。

    云端镜像通常不带本地模型权重，此时按 RRF 得分排序，保证检索链路可用。
    """
    if not config.reranker.enabled:
        logger.info("Reranker disabled by config, using RRF-score ordering")
        return ScoreReranker()
    import os
    if not os.path.isdir(BGEReranker.LOCAL_PATH):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            logger.warning("sentence_transformers 不可用且无本地重排模型，降级为 RRF 得分排序")
            return ScoreReranker()
    try:
        return BGEReranker(model_name=config.reranker.model_name, device=config.reranker.device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker model load failed (%s), fallback to RRF-score ordering", exc)
        return ScoreReranker()
