"""
实验设计 Agent（对齐《项目文档》5.2 示例与 5.5 的校验要求）

流程（两阶段）：
1. 生成阶段：基于检索证据生成实验方案；
2. 校验阶段：检查方案是否缺少对照组、重复数、终点指标等（可验证步骤），
   不合格则标注风险，而不是静默输出"看似可行"的方案。
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from agents.base import BaseAgent, build_evidence_context

logger = logging.getLogger(__name__)

DESIGN_PROMPT = """你是医疗科研平台的实验设计专家。请基于 <evidence> 的证据设计一个实验方案。
要求：
1. 明确实验目的与假设（把实验假设与临床结论分开表述）；
2. 方案要素齐全：实验对象/模型、分组（含对照组）、干预/处理条件、样本量或重复数、
   检测指标与终点指标、时间点、统计分析方法；
3. 每个设计依据标注引用编号；证据不足的设计点明确说明"基于常规设计，证据不足"；
4. 中文回答，结构清晰。

<query>
{query}
</query>

<evidence>
{context}
</evidence>"""

VALIDATION_PROMPT = """你是实验方案评审专家。请检查下面的实验方案：
1. 是否包含对照组？缺失则指出；
2. 是否说明重复数/样本量？缺失则指出；
3. 是否定义了明确的终点指标？缺失则指出；
4. 是否说明统计分析方法？
5. 是否存在无法执行的步骤？

只输出 JSON：{{"passed": true/false, "issues": ["问题1", "问题2"]}}

<protocol>
{protocol}
</protocol>"""


class ExperimentAgent(BaseAgent):
    name = "experiment_agent"

    def __init__(self, config, llm_factory, retriever, query_processor):
        super().__init__(config, llm_factory)
        self.retriever = retriever
        self.query_processor = query_processor

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        queries = self.query_processor.build_queries(query)
        chunks = self.retriever.retrieve(queries, tenant_id=state.get("tenant_id", "mihc"))
        context, citations = build_evidence_context(chunks, self.config.retrieval.max_context_chars)

        if not chunks:
            return {
                "output": "知识库中没有支持实验设计的证据，无法生成有依据的实验方案。请先补充相关文献或知识库资料。",
                "citations": [],
                "risks": [{"level": "warning", "message": "实验设计缺乏证据支持"}],
            }

        # 阶段 1：生成方案
        protocol = self.llm_factory.invoke_with_failover(
            [{"role": "user", "content": DESIGN_PROMPT.format(query=query, context=context)}],
            role="experiment_design",
        )

        # 阶段 2：方案校验（对照组/重复数/终点指标）
        issues = []
        try:
            import json
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": VALIDATION_PROMPT.format(protocol=protocol)}],
                role="citation_check",
            )
            raw = raw.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            result = json.loads(raw)
            issues = result.get("issues", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Protocol validation failed: %s", exc)
            issues = ["方案校验未完成，请人工复核"]

        risks = [
            {"level": "warning", "message": "实验方案为科研设计建议，实施前请经伦理审查与专业复核"},
        ]
        if issues:
            protocol += "\n\n## ⚠ 方案校验发现的问题\n" + "\n".join(f"- {i}" for i in issues)
            risks.append({"level": "warning", "message": f"校验发现 {len(issues)} 个问题，已列在方案末尾"})

        return {
            "output": protocol,
            "citations": [c.model_dump() for c in citations],
            "risks": risks,
        }
