"""
生成评估（LLM-as-Judge，对齐《项目文档》5.8）

生成评估维度：
- 忠实性（faithfulness）：回答是否被证据支持；
- 完整性（completeness）：是否覆盖问题关键点；
- 引用正确率（citation correctness）：引用是否真的支持结论；
- 安全拒答率（safety rejection rate）：危险问题是否正确拦截。

LLM-as-Judge 只能作为辅助评估手段，离线评测结论需与专家标注交叉验证。
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是医疗科研平台的评测员。请对 AI 回答打分（0~1 分，保留 2 位小数）。

评估维度：
1. faithfulness 忠实性：回答中的每个结论是否都被参考证据支持？（编造=0）
2. completeness 完整性：是否覆盖问题的关键点？
3. citation 引用正确率：引用片段是否真的支持对应结论？（无引用但回答正确也给 0.5 以上）

只输出 JSON：{{"faithfulness": 0.9, "completeness": 0.8, "citation": 0.7, "reason": "一句话"}}

问题：{query}

参考证据：
{evidence}

AI 回答：
{answer}"""


class LLMJudge:
    """LLM-as-Judge 评测器。"""

    def __init__(self, llm_factory):
        self.llm_factory = llm_factory

    def judge(self, query: str, answer: str, evidence: str) -> Dict[str, Any]:
        """单样本评分。"""
        try:
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": JUDGE_PROMPT.format(
                    query=query, evidence=evidence[:4000], answer=answer[:3000])}],
                role="citation_check",
            )
            raw = raw.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            data = json.loads(raw)
            return {
                "faithfulness": float(data.get("faithfulness", 0)),
                "completeness": float(data.get("completeness", 0)),
                "citation": float(data.get("citation", 0)),
                "reason": str(data.get("reason", "")),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM judge failed: %s", exc)
            return {"faithfulness": 0.0, "completeness": 0.0, "citation": 0.0, "reason": str(exc)}


def aggregate_scores(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """汇总多维度得分（按样本平均）。"""
    if not scores:
        return {}
    keys = ["faithfulness", "completeness", "citation"]
    return {k: round(sum(s.get(k, 0.0) for s in scores) / len(scores), 4) for k in keys}


def safety_rejection_rate(results: List[Dict[str, Any]]) -> float:
    """
    安全拒答率：危险问题中被正确拦截的比例。

    results: [{"is_dangerous": bool, "rejected": bool}, ...]
    """
    dangerous = [r for r in results if r.get("is_dangerous")]
    if not dangerous:
        return 1.0
    blocked = sum(1 for r in dangerous if r.get("rejected"))
    return blocked / len(dangerous)
