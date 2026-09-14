"""
数据分析 Agent（对齐平台目标"分析科研数据"）

职责：
- 若问题中带有具体数值数据 → 调用安全统计工具（describe/ttest/pearson）计算并解读；
- 否则 → 检索知识库中统计方法证据，给出方法学建议（含引用）；
- 输出附统计解释与科研风险提示。
"""

from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any

from agents.base import BaseAgent
from agents.tools import ToolRegistry

logger = logging.getLogger(__name__)

# 让 LLM 决定调用哪个工具及参数（Function-Calling 风格，但由我们做参数校验与超时）
TOOL_DECISION_PROMPT = """你是数据分析工具调用决策器。用户提供了一组数据，请决定调用哪个统计工具。
可用工具：
- describe：描述统计（均值/标准差/中位数/极值），参数 data=数值列表
- ttest_ind：两独立样本 t 检验，参数 a、b 两个数值列表
- ttest_paired：配对样本 t 检验，参数 a、b（等长）
- pearson：Pearson 相关分析，参数 a、b（等长）
- none：不需要工具

只输出 JSON：{{"tool": "describe", "a": [1,2,3], "b": []}} 或 {{"tool": "none"}}

用户输入：{query}"""

INTERPRET_PROMPT = """你是医疗科研平台的数据分析专家。基于工具计算结果给出解释：
1. 用通俗语言解释统计结果含义（均值差异、p 值意义、相关性方向与强度）；
2. 说明统计学显著不等于生物学/临床显著；
3. 提示局限：样本量、分布假设、多重比较等；
4. 医疗建议与科研建议分开表达；
5. 中文回答，不编造数据。

问题：{query}
工具结果：{tool_result}"""

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
        self.tools = ToolRegistry(timeout_sec=config.platform.tool_timeout_sec)

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        has_data = re.search(r"\d+(?:\.\d+)?", query) is not None

        if has_data:
            return self._run_with_tool(query)
        return self._recommend_method(state, query)

    # ---- 数据 + 工具路径 ----
    def _run_with_tool(self, query: str) -> Dict[str, Any]:
        try:
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": TOOL_DECISION_PROMPT.format(query=query)}],
                role="decision",
            )
            raw = raw.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            decision = json.loads(raw)
            tool = decision.get("tool", "none")

            if tool == "none":
                raise ValueError("no tool needed")

            tool_result = self.tools.run(
                tool,
                data=decision.get("a", decision.get("data", [])),
                a=decision.get("a", []),
                b=decision.get("b", []),
            )
        except Exception as exc:  # noqa: BLE001
            # 参数校验失败/超时/无工具 → 给出方法学建议而非猜测
            logger.warning("Tool path failed (%s), fallback to method suggestion", exc)
            return {
                "output": f"未能对您的数据执行统计计算（原因：{exc}）。建议先明确数据类型与分组方式，"
                          f"或提供完整的数值列表后重试。",
                "citations": [],
                "risks": [{"level": "warning", "message": "统计计算未执行"}],
            }

        answer = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": INTERPRET_PROMPT.format(query=query, tool_result=tool_result)}],
            role="data_analysis",
        )
        return {
            "output": answer,
            "citations": [],
            "risks": [
                {"level": "warning", "message": "统计结果仅反映输入数据本身，需结合实验设计与生物学意义解读"},
                {"level": "info", "message": f"已执行工具: {decision.get('tool', '')}"},
            ],
        }

    # ---- 方法推荐路径（检索知识库）----
    def _recommend_method(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        from agents.base import build_evidence_context
        queries = self.query_processor.build_queries(f"医学统计方法 {query}")
        chunks = self.retriever.retrieve(queries, tenant_id=state.get("tenant_id", "mihc"))
        context, citations = build_evidence_context(chunks, self.config.retrieval.max_context_chars)
        if not chunks:
            return {
                "output": "知识库中暂无相关统计方法资料，无法给出有依据的方法推荐。",
                "citations": [],
                "risks": [{"level": "warning", "message": "统计方法证据缺失"}],
            }
        answer = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": METHOD_PROMPT.format(query=query, context=context)}],
            role="data_analysis",
        )
        return {
            "output": answer,
            "citations": [c.model_dump() for c in citations],
            "risks": [{"level": "info", "message": "方法推荐仅基于知识库证据，实际分析请结合数据分布验证"}],
        }
