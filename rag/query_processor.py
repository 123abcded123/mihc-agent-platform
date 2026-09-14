"""
查询处理模块：Query Rewrite + HyDE（对齐《项目文档》5.4 第一步）

原则（文档明确要求）：
- 保留原始问题，同时生成适合检索的关键词和扩展问题；
- 不要只保留改写结果，因为改写模型也可能误解用户意图；
- HyDE 生成"假设文档"，用其向量补充语义召回。
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是一名医疗科研检索专家。请把用户问题改写成更适合检索的形式：
1. 保留核心医学术语，并补充同义表达（中英文术语、缩略词）；
2. 输出不超过 3 条改写查询，每条一行；
3. 不要改变原问题的意图，不要编造不存在的内容。

用户问题：{query}

改写查询（每行一条）："""

HYDE_PROMPT = """你是一名医疗科研专家。请根据用户问题，写一段 200 字以内的"假设性答案文档"（hypothetical document）。
要求：
1. 内容围绕问题的核心概念展开（如机制、检测方法、研究现状）；
2. 可以基于一般医学知识推断，但不要给出确定性临床结论；
3. 直接输出文档正文，不要任何前缀。

用户问题：{query}"""


class QueryProcessor:
    """查询改写与 HyDE 生成。"""

    def __init__(self, config, llm_factory):
        self.config = config
        self.llm_factory = llm_factory

    def rewrite(self, query: str) -> List[str]:
        """LLM 查询改写：返回最多 3 条改写查询（不含原始问题）。"""
        if self.config.platform.fast_mode or not self.config.retrieval.enable_rewrite:
            return []
        try:
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
                role="rewrite",
            )
            lines = [ln.strip(" -•·") for ln in raw.splitlines() if ln.strip()]
            return [ln for ln in lines if ln][:3]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed: %s", exc)
            return []

    def hyde(self, query: str) -> str:
        """生成假设文档；失败时返回空串。"""
        if self.config.platform.fast_mode or not self.config.retrieval.enable_hyde:
            return ""
        try:
            return self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
                role="hyde",
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE generation failed: %s", exc)
            return ""

    def build_queries(self, query: str) -> List[str]:
        """
        构建检索查询列表：原始问题 + 改写查询 + HyDE 文档。
        原始问题始终保留（改写模型可能误解意图）。
        """
        queries: List[str] = [query]
        for rw in self.rewrite(query):
            if rw and rw != query:
                queries.append(rw)
        hyde_doc = self.hyde(query)
        if hyde_doc:
            queries.append(hyde_doc)
        return queries
