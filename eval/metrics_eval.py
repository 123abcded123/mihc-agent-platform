
from __future__ import annotations

from typing import List, Dict, Any

def hit_rate_at_k(recalled_ids: List[List[str]], relevant_ids: List[List[str]], k: int) -> float:
    if not recalled_ids:
        return 0.0
    hits = 0
    for recalled, relevant in zip(recalled_ids, relevant_ids):
        if set(recalled[:k]) & set(relevant):
            hits += 1
    return hits / len(recalled_ids)

def recall_at_k(recalled_ids: List[List[str]], relevant_ids: List[List[str]], k: int) -> float:
    if not recalled_ids:
        return 0.0
    total = 0.0
    for recalled, relevant in zip(recalled_ids, relevant_ids):
        if not relevant:
            continue
        total += len(set(recalled[:k]) & set(relevant)) / len(set(relevant))
    return total / len(recalled_ids)

def mrr(recalled_ids: List[List[str]], relevant_ids: List[List[str]]) -> float:
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
    recalled = [r["recalled_ids"] for r in results]
    relevant = [r["relevant_ids"] for r in results]
    metrics: Dict[str, Any] = {"mrr": round(mrr(recalled, relevant), 4), "total_samples": len(results)}
    for k in k_list:
        metrics[f"hit_rate@{k}"] = round(hit_rate_at_k(recalled, relevant, k), 4)
        metrics[f"recall@{k}"] = round(recall_at_k(recalled, relevant, k), 4)
    return metrics
