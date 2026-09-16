
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_sft_data(path: str) -> List[Dict[str, str]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def format_prompt(sample: Dict[str, Any]) -> str:
    instruction = sample["instruction"]
    inp = sample.get("input", "")
    if inp:
        return f"{instruction}\n\n{inp}"
    return instruction

def main():
    parser = argparse.ArgumentParser(description="Qwen3 LoRA SFT")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./models/qwen3-32b-mihc-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", help="QLoRA 4bit 量化（显存不足时开启）")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DataCollatorForCompletionOnlyLM

    logger.info("加载模型: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "device_map": "auto"}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = load_sft_data(args.data)
    logger.info("训练样本: %d", len(samples))
    formatted = [{"prompt": format_prompt(s), "answer": s["output"]} for s in samples]

    def tokenize_fn(example):
        full = f"### 指令:\n{example['prompt']}\n\n### 回答:\n{example['answer']}"
        out = tokenizer(full, truncation=True, max_length=args.max_length)
        out["labels"] = out["input_ids"].copy()
        return out

    from datasets import Dataset
    dataset = Dataset.from_list(formatted).map(tokenize_fn, remove_columns=["prompt", "answer"])

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("LoRA 权重已保存: %s", args.output)

if __name__ == "__main__":
    main()
