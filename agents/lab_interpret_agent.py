"""
检验解读 Agent：解读检查检验结果（对齐平台目标"理解检查检验结果"）

职责：
- 从知识库检索临床检验标准/指标参考范围证据；
- 结合用户提供的检验数据做解读；
- 强制输出安全提示（检验解读≠诊断，需结合临床）。
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from agents.base import BaseAgent, generate_with_evidence

logger = logging.getLogger(__name__)

LAB_RULES = """你是医疗科研平台的检验解读 Agent。
解读规则：
1. 仅依据 <evidence> 中的检验标准/参考范围解读，缺失的指标参考范围必须明确说明；
2. 对超出参考范围的指标：说明可能原因（区分常见生理性/病理性因素）；
3. 医疗建议与科研建议分开表达；不得给出确定性诊断结论；
4. 结尾必须附"建议结合临床医生面诊"的安全提示；
5. 中文回答。"""


class LabInterpretAgent(BaseAgent):
    name = "lab_agent"

    def __init__(self, config, llm_factory, retriever, query_processor):
        super().__init__(config, llm_factory)
        self.retriever = retriever
        self.query_processor = query_processor

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        # 检验解读检索侧重"标准/参考范围/指标"，改写查询时追加引导词
        queries = self.query_processor.build_queries(f"检验指标解读 参考范围 {query}")
        chunks = self.retriever.retrieve(
            queries,
            tenant_id=state.get("tenant_id", "mihc"),
        )

        context_parts = []
        from agents.base import build_evidence_context
        context, citations = build_evidence_context(chunks, self.config.retrieval.max_context_chars)

        if not chunks:
            return {
                "output": "知识库中暂无该检验指标的参考标准证据，无法给出有依据的解读。建议补充临床检验标准文档。",
                "citations": [],
                "risks": [{"level": "warning", "message": "缺少检验标准证据，解读未执行"}],
            }

        prompt = f"""{LAB_RULES}

<query>
{query}
</query>

<evidence>
{context}
</evidence>

请解读："""
        answer = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": prompt}],
            role="rag_generate",
        )
        return {
            "output": answer,
            "citations": [c.model_dump() for c in citations],
            "risks": [
                {"level": "warning", "message": "检验解读仅用于科研参考，最终诊断请结合临床医生意见"},
            ],
        }
