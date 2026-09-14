"""
DPO 训练脚本（对齐《项目文档》5.6：用偏好对做 DPO）

SFT 学"怎样回答"，DPO 学"两个回答哪个更符合偏好"。
DPO 不是补充知识的唯一办法；知识更新更适合更新知识库。

用法：
  python finetune/dpo_train.py --model ./models/qwen3-32b-mihc-lora \
      --data ./data/instruction_dataset/dpo_train.jsonl \
      --output ./models/qwen3-32b-mihc-dpo
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dpo_data(path: str) -> List[Dict[str, str]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="DPO 偏好对齐")
    parser.add_argument("--model", default="./models/qwen3-32b-mihc-lora")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./models/qwen3-32b-mihc-dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta（KL 约束强度）")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer, DPOConfig

    logger.info("加载基座（SFT 后）模型: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    for p in ref_model.parameters():
        p.requires_grad = False

    samples = load_dpo_data(args.data)
    logger.info("偏好对样本: %d", len(samples))

    from datasets import Dataset
    dataset = Dataset.from_list([
        {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
        for s in samples
    ])

    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        bf16=True,
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=1024,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("DPO 权重已保存: %s", args.output)


if __name__ == "__main__":
    main()
