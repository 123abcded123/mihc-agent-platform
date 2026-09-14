# 微调模块（对齐《项目文档》5.6）

1.5 万条医疗科研指令数据 → LoRA 微调 + DPO 强化学习，答准率 62% → 98%。

## 流程

```text
收集历史问题 → 去重 → 脱敏 → 专家审核答案
→ 标注意图/难度/来源/安全等级 → 划分训练/验证/测试
→ SFT/LoRA → DPO → 离线评测 → 小流量上线
```

## 脚本

| 脚本 | 用途 | 产出 |
|---|---|---|
| `build_dataset.py` | 去重/脱敏/标注/切分（训练测试隔离） | `data/instruction_dataset/*.jsonl` |
| `sft_lora.py` | Qwen3 LoRA SFT（支持 QLoRA 4bit） | LoRA adapter |
| `dpo_train.py` | DPO 偏好对齐 | DPO 模型 |
| `train_intent.py` | BERT 意图分类微调（93.5%+ 准确率） | `INTENT_MODEL_PATH` |

## 依赖

```bash
pip install peft trl datasets bitsandbytes accelerate
```

## 示例

```bash
# 1. 构建数据集（训练/验证/测试隔离）
python finetune/build_dataset.py --input ./data/raw_qa.jsonl --out_dir ./data/instruction_dataset --dpo

# 2. LoRA SFT（需 GPU；32B 建议 QLoRA）
python finetune/sft_lora.py --model Qwen/Qwen3-32B --data ./data/instruction_dataset/sft_train.jsonl --use_4bit

# 3. DPO（用偏好对数据）
python finetune/dpo_train.py --model ./models/qwen3-32b-mihc-lora --data ./data/instruction_dataset/dpo_train.jsonl

# 4. BERT 意图微调（CPU 可训练小模型）
python finetune/train_intent.py --data ./data/intent_dataset.jsonl --output ./models/intent_bert
```

## 部署（vLLM）

```bash
# LoRA 合并或动态加载后以 vLLM 提供 OpenAI 兼容服务
vllm serve Qwen/Qwen3-32B --enable-lora --lora-modules mihc=./models/qwen3-32b-mihc-dpo
# 平台 .env 配置：LLM_BASE_URL=http://<vllm-host>:8000/v1  LLM_MODEL=mihc
```

## 注意

- **训练/评测隔离**：test 集禁止参与训练与调参（文档 5.6：62%→98% 可能只是记住了答案）；
- DPO 只调偏好，知识更新优先更新知识库；
- 本机 RTX 3050 4GB 无法训练 32B 模型，脚本面向 GPU 训练环境。
