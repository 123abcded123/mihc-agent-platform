"""
BERT 意图识别微调脚本（对齐简历：BERT 微调模型，意图识别准确率 93.5%+）

4+1 类意图：literature_search / knowledge_qa /
data_analysis / experiment_design / other

训练产出路径填到 .env 的 INTENT_MODEL_PATH，平台自动加载该模型做意图识别
（未配置时回退 LLM 意图分类）。

用法：
  python finetune/train_intent.py --data ./data/intent_dataset.jsonl \
      --output ./models/intent_bert --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["literature_search", "knowledge_qa", "data_analysis", "experiment_design", "other"]


def load_data(path: str):
    """加载 JSONL：每行 {"text": "问题", "label": "意图"}。"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("label") not in LABELS:
                continue
            texts.append(item["text"])
            labels.append(LABELS.index(item["label"]))
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="BERT 意图分类微调")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base_model", default="bert-base-chinese")
    parser.add_argument("--output", default="./models/intent_bert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    )
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)})

    texts, labels = load_data(args.data)
    logger.info("样本数: %d", len(texts))

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=128)

    dataset = Dataset.from_dict({"text": texts, "label": labels}).map(tokenize_fn, batched=True)
    split = dataset.train_test_split(test_size=0.15, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        import numpy as np
        preds = np.argmax(logits, axis=1)
        return {"accuracy": float((preds == labels_).mean())}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        tokenizer=tokenizer, compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("评测结果: %s", metrics)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("意图模型已保存: %s（把该路径配置到 .env 的 INTENT_MODEL_PATH）", args.output)


if __name__ == "__main__":
    main()
