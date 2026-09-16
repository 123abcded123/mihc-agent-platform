
from __future__ import annotations

import logging
from typing import Dict, Any

from agents.base import BaseAgent, build_evidence_context

logger = logging.getLogger(__name__)

METHOD_PROMPT = """你是医疗科研平台的数据分析专家。请基于 <evidence> 中的统计方法资料，
为用户的数据分析问题推荐合适的方法：
1. 给出方法选择理由（数据类型、组数、分布假设）；
2. 说明前提假设与常见误用；
3. 每个关键结论标注引用编号；证据不足时明确说明。
中文回答。

<query>
{query}
</query>

<evidence>
{context}
</evidence>"""

class DataAnalysisAgent(BaseAgent):
    name = "data_analysis_agent"

    def __init__(self, config, llm_factory, retriever, query_processor):
        super().__init__(config, llm_factory)
        self.retriever = retriever
        self.query_processor = query_processor

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        queries = self.query_processor.build_queries(f"医学统计方法 {query}")
        chunks = self.retriever.retrieve(queries, tenant_id=state.get("tenant_id", "mihc"))
        if not chunks:
            return {
                "output": "知识库中暂无相关统计方法资料，无法给出有依据的方法推荐。",
                "citations": [],
                "risks": [{"level": "warning", "message": "统计方法证据缺失"}],
            }
        context, citations = build_evidence_context(chunks, self.config.retrieval.max_context_chars)
        answer = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": METHOD_PROMPT.format(query=query, context=context)}],
            role="data_analysis",
        )
        return {
            "output": answer,
            "citations": [c.model_dump() for c in citations],
            "risks": [
                {"level": "info", "message": "方法推荐仅基于知识库证据；客户表格统计请到项目工作台执行，由确定性工具计算"},
            ],
        }
