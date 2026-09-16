
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRIVACY_PATTERNS = [
    r"\b1[3-9]\d{9}\b",
    r"\b\d{17}[\dXx]\b",
    r"(?<=姓|叫|患者为)\S{1,4}(?=，|。|的)",
    r"\b[\w.-]+@[\w.-]+\.\w+\b",
]

INTENT_TAGS = ["literature_search", "knowledge_qa", "data_analysis", "experiment_design", "other"]
DIFFICULTY_TAGS = ["easy", "medium", "hard"]
SAFETY_TAGS = ["safe", "sensitive", "reject"]

def deduplicate(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for s in samples:
        key = hashlib.md5(s["query"].strip().lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def remove_privacy(text: str) -> str:
    for pattern in PRIVACY_PATTERNS:
        text = re.sub(pattern, "[已脱敏]", text)
    return text

def normalize(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample["query"] = remove_privacy(sample["query"].strip())
    sample["answer"] = remove_privacy(sample.get("answer", "").strip())
    if "intent" not in sample or sample["intent"] not in INTENT_TAGS:
        sample["intent"] = "other"
    if "difficulty" not in sample or sample["difficulty"] not in DIFFICULTY_TAGS:
        sample["difficulty"] = "easy"
    if "source" not in sample:
        sample["source"] = "collected"
    if "safety" not in sample or sample["safety"] not in SAFETY_TAGS:
        sample["safety"] = "safe" if sample["intent"] != "reject" else "reject"
    return sample

def split_datasets(samples: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)
    data = list(samples)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
    }

def to_sft_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instruction": sample["query"],
        "input": "",
        "output": sample["answer"],
        "metadata": {
            "intent": sample["intent"],
            "difficulty": sample["difficulty"],
            "safety": sample["safety"],
            "source": sample["source"],
        },
    }

def to_dpo_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": sample["query"],
        "chosen": sample.get("answer", ""),
        "rejected": sample.get("rejected_answer", sample.get("answer", "")),
        "metadata": {"intent": sample["intent"]},
    }

def main():
    parser = argparse.ArgumentParser(description="构建 SFT/DPO 指令数据集")
    parser.add_argument("--input", required=True, help="原始数据 JSONL（每行 {query, answer?, intent?, ...}）")
    parser.add_argument("--out_dir", default="./data/instruction_dataset")
    parser.add_argument("--dpo", action="store_true", help="同时生成 DPO 偏好对数据（需要 rejected_answer 字段）")
    args = parser.parse_args()

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info("原始样本: %d", len(samples))
    samples = deduplicate(samples)
    logger.info("去重后: %d", len(samples))
    samples = [normalize(s) for s in samples]
    splits = split_datasets(samples)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in splits.items():
        path = os.path.join(args.out_dir, f"sft_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in part:
                f.write(json.dumps(to_sft_format(s), ensure_ascii=False) + "\n")
        logger.info("SFT %s: %d -> %s", name, len(part), path)

    if args.dpo:
        for name, part in splits.items():
            path = os.path.join(args.out_dir, f"dpo_{name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for s in part:
                    if s.get("rejected_answer"):
                        f.write(json.dumps(to_dpo_format(s), ensure_ascii=False) + "\n")
            logger.info("DPO %s -> %s", name, path)

    logger.info("完成：训练/验证/测试已隔离，禁止用 test 集参与训练")

if __name__ == "__main__":
    main()
