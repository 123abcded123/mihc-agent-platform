
from __future__ import annotations

import logging
from typing import Dict, Any, List

from agents.base import BaseAgent, build_evidence_context, generate_with_evidence

logger = logging.getLogger(__name__)

LITERATURE_PROMPT = """你是医疗科研平台的文献检索 Agent。
请基于检索到的证据整理文献综述式回答：
1. 先总结各证据片段的研究内容（年份、方法、结论）；
2. 按用户问题的条件筛选（如年份、模型、实验对象）；
3. 最后给出综合结论，并指出证据空白。
要求：每个结论标注引用编号；只使用给出的证据；中文回答。"""

class LiteratureAgent(BaseAgent):
    name = "literature_agent"

    def __init__(self, config, llm_factory, retriever, query_processor):
        super().__init__(config, llm_factory)
        self.retriever = retriever
        self.query_processor = query_processor

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        queries = self.query_processor.build_queries(query)
        chunks = self.retriever.retrieve(
            queries,
            tenant_id=state.get("tenant_id", "mihc"),
        )
        if not chunks:
            return {
                "output": "未检索到相关文献证据，请调整检索条件或补充知识库。",
                "citations": [],
                "risks": [{"level": "info", "message": "文献检索无结果"}],
            }

        context, citations = build_evidence_context(chunks, self.config.retrieval.max_context_chars)
        prompt = f"""{LITERATURE_PROMPT}

<query>
{query}
</query>

<evidence>
{context}
</evidence>

请作答："""
        answer = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": prompt}],
            role="rag_generate",
        )
        return {
            "output": answer,
            "citations": [c.model_dump() for c in citations],
            "risks": [{"level": "info", "message": "文献结论仅基于当前知识库证据，实验假设不构成临床建议"}],
        }
