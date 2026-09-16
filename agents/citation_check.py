
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

CHECK_PROMPT = """你是引用审查员。检查下面的回答：
1. 回答中标注的每个引用编号 [N] 是否都出现在 <evidence> 中？
2. 关键结论是否真的被对应引用片段支持？

只输出 JSON：{{"missing_refs": [2, 5], "unsupported": ["结论X缺少支持"]}}
没有问题时输出：{{"missing_refs": [], "unsupported": []}}

<answer>
{answer}
</answer>

<evidence>
{evidence}
</evidence>"""

def check_citations(state: Dict[str, Any], llm_factory, fast_mode: bool = False) -> Dict[str, Any]:
    answer = state.get("merged_output", "")
    citations = state.get("citations", [])
    risks = state.get("risks", [])

    if not citations:
        logger.info("Citation check: no citations, skip")
        return {"answer": answer, "citations": citations, "risks": risks}

    refs = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    out_of_range = sorted(r for r in refs if r < 1 or r > len(citations))
    if out_of_range:
        risks.append({"level": "warning", "message": f"引用编号越界: {out_of_range}，已标注请人工复核"})
        answer += f"\n\n> ⚠ 引用检查：编号 {out_of_range} 超出证据范围。"

    if fast_mode:
        return {"answer": answer, "citations": citations, "risks": risks}

    try:
        evidence = "\n\n".join(f"[{i}] {c.get('text_snippet', '')[:300]}"
                               for i, c in enumerate(citations, start=1))
        raw = llm_factory.invoke_with_failover(
            [{"role": "user", "content": CHECK_PROMPT.format(answer=answer[:3000], evidence=evidence[:4000])}],
            role="citation_check",
        )
        import json
        raw = raw.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        result = json.loads(raw)
        missing = result.get("missing_refs", [])
        unsupported = result.get("unsupported", [])
        if missing or unsupported:
            msg = "引用检查发现问题：" + "; ".join(
                ([f"缺失引用{missing}"] if missing else []) +
                ([f"结论缺支持:{unsupported}"] if unsupported else [])
            )
            risks.append({"level": "warning", "message": msg})
            answer += f"\n\n> ⚠ 引用检查：{msg}"
    except Exception as exc:
        logger.warning("LLM citation check failed: %s", exc)

    return {"answer": answer, "citations": citations, "risks": risks}
