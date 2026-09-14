"""
Harness 评测指标计算（对齐《项目文档》5.8 检索评估）

指标：
- HitRate@K：正确答案是否出现在前 K 个召回结果中；
- Recall@K：命中的相关片段占全部相关片段的比例；
- MRR：第一个相关片段排名的倒数均值。

生成评估指标（LLM-as-Judge）在 judge.py 中实现。
"""

from __future__ import annotations

from typing import List, Dict, Any


def hit_rate_at_k(recalled_ids: List[List[str]], relevant_ids: List[List[str]], k: int) -> float:
    """HitRate@K：前 K 个召回结果中至少命中一个相关片段的问题比例。"""
    if not recalled_ids:
        return 0.0
    hits = 0
    for recalled, relevant in zip(recalled_ids, relevant_ids):
        if set(recalled[:k]) & set(relevant):
            hits += 1
    return hits / len(recalled_ids)


def recall_at_k(recalled_ids: List[List[str]], relevant_ids: List[List[str]], k: int) -> float:
    """Recall@K：命中的相关片段占全部相关片段的比例（按问题平均）。"""
    if not recalled_ids:
        return 0.0
    total = 0.0
    for recalled, relevant in zip(recalled_ids, relevant_ids):
        if not relevant:
            continue
        total += len(set(recalled[:k]) & set(relevant)) / len(set(relevant))
    return total / len(recalled_ids)


def mrr(recalled_ids: List[List[str]], relevant_ids: List[List[str]]) -> float:
    """MRR：第一个相关片段排名的倒数均值。"""
    if not recalled_ids:
        return 0.0
    total = 0.0
    for recalled, relevant in zip(recalled_ids, relevant_ids):
        rr = 0.0
        for rank, cid in enumerate(recalled, start=1):
            if cid in set(relevant):
                rr = 1.0 / rank
                break
        total += rr
    return total / len(recalled_ids)


def compute_retrieval_metrics(results: List[Dict[str, Any]], k_list: List[int] = (1, 3, 5)) -> Dict[str, Any]:
    """
    计算检索指标。

    Args:
        results: [{"recalled_ids": [...], "relevant_ids": [...]}, ...]
    Returns:
        {"hit_rate@{k}", "recall@{k}", "mrr", "total_samples"}
    """
    recalled = [r["recalled_ids"] for r in results]
    relevant = [r["relevant_ids"] for r in results]
    metrics: Dict[str, Any] = {"mrr": round(mrr(recalled, relevant), 4), "total_samples": len(results)}
    for k in k_list:
        metrics[f"hit_rate@{k}"] = round(hit_rate_at_k(recalled, relevant, k), 4)
        metrics[f"recall@{k}"] = round(recall_at_k(recalled, relevant, k), 4)
    return metrics
