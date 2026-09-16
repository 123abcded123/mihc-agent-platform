
from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def merge_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
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

        for c in result.get("citations", []):
            key = c.get("chunk_id", "")
            if key and key not in seen_citations:
                seen_citations.add(key)
                citations.append(c)
            elif not key:
                citations.append(c)

        for r in result.get("risks", []):
            risks.append({"level": r.get("level", "info"), "message": r.get("message", "")})

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
