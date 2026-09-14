"""
结果合并模块（对齐《项目文档》5.5：结果合并 → 引用检查）

职责：把多 Agent 的中间产物（上下文隔离在 agent_outputs）合并成最终回答草稿：
- 合并顺序按规划步骤；
- 去重引用（按 chunk_id）；
- 汇总风险提示（按级别排序）。
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def merge_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """合并所有步骤产物为 {merged_output, citations, risks}。"""
    outputs = state.get("agent_outputs", {})
    plan = state.get("plan", [])

    sections: List[str] = []
    citations: List[Dict[str, Any]] = []
    seen_citations = set()
    risks: List[Dict[str, str]] = []

    for step in plan:
        step_id = str(step.get("step_id", ""))
        result = outputs.get(step_id)
        if not result:
            continue
        agent_name = step.get("agent", "agent")
        text = result.get("output", "")
        if text:
            sections.append(f"### {agent_name}\n{text}")

        # 引用去重
        for c in result.get("citations", []):
            key = c.get("chunk_id", "")
            if key and key not in seen_citations:
                seen_citations.add(key)
                citations.append(c)
            elif not key:  # 无 chunk_id 的引用（如工具结果）也保留
                citations.append(c)

        for r in result.get("risks", []):
            risks.append({"level": r.get("level", "info"), "message": r.get("message", "")})

    # 风险排序：critical > warning > info
    level_order = {"critical": 0, "warning": 1, "info": 2}
    risks = sorted(risks, key=lambda r: level_order.get(r["level"], 3))

    if not sections:
        merged = "未找到可合并的结果。"
    elif len(sections) == 1:
        merged = sections[0]
    else:
        merged = "\n\n".join(sections)

    logger.info("Merge: %d sections, %d citations, %d risks", len(sections), len(citations), len(risks))
    return {"merged_output": merged, "citations": citations, "risks": risks}
