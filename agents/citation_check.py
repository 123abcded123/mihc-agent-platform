"""
引用检查模块（对齐《项目文档》5.8：引用正确率——引用是否真的支持结论）

职责：抽查最终回答中的引用编号是否真实存在于检索证据中，
并用 LLM 检查关键结论与引用片段的支撑关系。
"""

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
    """校验引用完整性；发现问题时在答案末尾附加说明。"""
    answer = state.get("merged_output", "")
    citations = state.get("citations", [])
    risks = state.get("risks", [])

    if not citations:
        logger.info("Citation check: no citations, skip")
        return {"answer": answer, "citations": citations, "risks": risks}

    # 快速静态检查：引用的编号是否有越界
    refs = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    out_of_range = sorted(r for r in refs if r < 1 or r > len(citations))
    if out_of_range:
        risks.append({"level": "warning", "message": f"引用编号越界: {out_of_range}，已标注请人工复核"})
        answer += f"\n\n> ⚠ 引用检查：编号 {out_of_range} 超出证据范围。"

    if fast_mode:
        return {"answer": answer, "citations": citations, "risks": risks}

    # LLM 深度检查（抽样：证据过长时截断）
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM citation check failed: %s", exc)

    return {"answer": answer, "citations": citations, "risks": risks}
