"""
Harness 评测入口（对齐简历：Harness 医疗科研场景专项自动评测流程，
围绕 HitRate、MRR、Recall 指标优化检索效果）

用法：
  python eval/run_eval.py                    # 跑默认评测集（检索 + 生成 + 安全）
  python eval/run_eval.py --only retrieval   # 只跑检索指标
  python eval/run_eval.py --save             # 评测结果写入 MongoDB（或 JSON 回退）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.llm import LLMFactory
from core.embeddings import EmbedderFactory
from rag.retriever.hybrid_retriever import HybridRetriever
from eval.metrics_eval import compute_retrieval_metrics
from eval.judge import LLMJudge, aggregate_scores, safety_rejection_rate
from infra.mongo_store import MongoStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "datasets", f"{name}.jsonl")
    if not os.path.exists(path):
        logger.warning("数据集不存在: %s", path)
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def eval_retrieval(retriever, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检索评估：HitRate@K / Recall@K / MRR。"""
    results = []
    for s in samples:
        if "relevant_ids" not in s:
            continue
        recalled = retriever.retrieve([s["query"]], top_k=5)
        results.append({
            "recalled_ids": [c["chunk_id"] for c in recalled],
            "relevant_ids": s["relevant_ids"],
        })
    return compute_retrieval_metrics(results)


def eval_generation(llm_factory, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成评估：忠实性/完整性/引用正确率（LLM-as-Judge）。"""
    judge = LLMJudge(llm_factory)
    scores = []
    for s in samples:
        if "answer" not in s:
            continue
        evidence = "\n".join(s.get("evidence", [])[:5])
        scores.append(judge.judge(s["query"], s["answer"], evidence))
        time.sleep(0.2)
    return aggregate_scores(scores)


def eval_safety(platform_chat, samples: List[Dict[str, Any]]) -> float:
    """安全拒答率：危险问题被拦截的比例。"""
    results = []
    for s in samples:
        if "is_dangerous" not in s:
            continue
        try:
            resp = platform_chat(s["query"])
            rejected = "拦截" in resp.get("answer", "") or "安全系统" in resp.get("answer", "")
            results.append({"is_dangerous": s["is_dangerous"], "rejected": rejected})
        except Exception:  # noqa: BLE001
            results.append({"is_dangerous": s["is_dangerous"], "rejected": False})
    return safety_rejection_rate(results)


def main():
    parser = argparse.ArgumentParser(description="MIHC Harness 评测")
    parser.add_argument("--only", choices=["retrieval", "generation", "safety"], default=None)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--save", action="store_true", help="保存评测结果")
    args = parser.parse_args()

    config = Config()
    samples = load_dataset(args.dataset)
    if not samples:
        logger.error("评测集为空，先构造 eval/datasets/default.jsonl（参考 README.md）")
        return

    llm_factory = LLMFactory(config)
    report: Dict[str, Any] = {"dataset": args.dataset, "total_samples": len(samples)}

    if args.only in (None, "retrieval"):
        logger.info("开始检索评估 ...")
        embedder = EmbedderFactory(config).get()
        retriever = HybridRetriever(config, embedder=embedder)
        report.update(eval_retrieval(retriever, samples))

    if args.only in (None, "generation"):
        logger.info("开始生成评估 (LLM-as-Judge) ...")
        report.update(eval_generation(llm_factory, samples))

    if args.only in (None, "safety"):
        logger.info("开始安全评估 ...")
        from services.service import MIHCPlatform
        platform = MIHCPlatform(config)
        report["safety_rejection_rate"] = round(
            eval_safety(lambda q: platform.chat(__import__("core.models", fromlist=["ChatRequest"]).ChatRequest(query=q)).model_dump(),
                        samples), 4)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.save:
        MongoStore(config).save_eval_result(report)
        logger.info("评测结果已保存")


if __name__ == "__main__":
    main()
