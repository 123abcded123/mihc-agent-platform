"""
BERT 意图识别模块（对齐简历：BERT 微调模型，意图识别准确率 93.5%+）

职责：对用户问题做 4+1 类意图分类：
  literature_search（文献检索）、knowledge_qa（知识问答）、
  data_analysis（数据分析）、experiment_design（实验设计）、other（其他）

实现：
- 优先加载微调后的本地 BERT 模型（INTENT_MODEL_PATH）；
- 无微调模型时返回 None，由上层回退 LLM 意图分类（同一标签集）；
- 微调脚本见 finetune/train_intent.py。
"""

from __future__ import annotations

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "literature_search",
    "knowledge_qa",
    "data_analysis",
    "experiment_design",
    "other",
]

# LLM 回退提示词（与 BERT 微调数据同一标签集，保证口径一致）
LLM_INTENT_PROMPT = """你是医疗科研平台的意图识别模块。请把用户问题归类到以下 5 类之一：
1. literature_search 文献检索：找论文、检索研究进展、需要引用来源（如"检索2020年后关于XX的研究"）
2. knowledge_qa 知识问答：询问医学知识、机制、定义（如"ROS 与氧化应激是什么关系"）
3. data_analysis 数据分析：科研数据处理、统计方法、图表分析（如"怎么比较两组表达差异"）
4. experiment_design 实验设计：设计实验方案、验证思路（如"设计一个体外实验验证XX"）
5. other 其他：闲聊、非医疗科研问题

只输出 JSON，格式：
{{"intent": "<类别>", "reasoning": "<一句话理由>", "confidence": 0.9}}

用户问题：{query}"""

# 规则意图关键词（快速模式/LLM 不可用时使用；与微调数据口径一致）
RULE_INTENT_KEYWORDS = {
    "literature_search": ["检索", "文献", "论文", "研究进展", "综述", "meta分析", "荟萃"],
    "experiment_design": ["实验设计", "实验方案", "设计一个实验", "体外实验", "体内实验", "验证实验"],
    "data_analysis": ["统计分析", "数据分析", "差异比较", "相关性", "回归", "聚类", "t检验", "方差分析"],
    "knowledge_qa": ["是什么", "为什么", "机制", "原理", "定义", "关系", "区别", "作用"],
}


def rule_based_intent(query: str) -> dict:
    """规则意图分类（快速模式）：关键词命中顺序即优先级。"""
    for intent, keywords in RULE_INTENT_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return {"intent": intent, "confidence": 0.8, "model": "rule"}
    return {"intent": "other", "confidence": 0.8, "model": "rule"}


class BERTIntentClassifier:
    """BERT 意图分类器（微调模型；未微调时不可用，返回 None）。"""

    def __init__(self, model_path: str, device: str = "cpu", labels: Optional[List[str]] = None):
        self.model_path = model_path
        self.device = device
        self.labels = labels or DEFAULT_LABELS
        self.model = None
        self.tokenizer = None
        self.available = False

        if model_path and os.path.isdir(model_path):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.to(device)
                self.model.eval()
                self.device_t = torch.device(device)
                # 若微调时写入 id2label 则以模型配置为准
                id2label = getattr(self.model.config, "id2label", None)
                if id2label:
                    self.labels = [id2label[i] for i in range(len(id2label))]
                self.available = True
                logger.info("BERT intent classifier loaded from %s (labels=%s)", model_path, self.labels)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BERT intent model load failed: %s", exc)

    def classify(self, query: str) -> Optional[dict]:
        """返回 {"intent", "confidence", "model": "bert"}；不可用时返回 None。"""
        if not self.available:
            return None
        import torch
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device_t) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
        return {
            "intent": self.labels[idx],
            "confidence": float(probs[idx].item()),
            "model": "bert",
        }


class IntentClassifier:
    """意图识别门面：BERT 优先，LLM 回退。"""

    def __init__(self, config, llm_factory):
        self.config = config
        self.llm_factory = llm_factory
        self.bert = BERTIntentClassifier(
            model_path=config.intent.model_path,
            device=config.intent.device,
            labels=config.intent.labels,
        )

    def classify(self, query: str) -> dict:
        """意图识别（含回退链路）。失败兜底 other，保证系统可用。

        优先级：微调 BERT > 规则（快速模式） > LLM 回退。
        """
        result = self.bert.classify(query)
        if result:
            return result

        if self.config.platform.fast_mode:
            return rule_based_intent(query)

        # LLM 回退（无微调 BERT 时）
        try:
            import json
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": LLM_INTENT_PROMPT.format(query=query)}],
                role="decision",
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            intent = data.get("intent", "other")
            if intent not in self.config.intent.labels:
                intent = "other"
            return {"intent": intent, "confidence": float(data.get("confidence", 0.9)), "model": "llm_fallback"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM intent fallback failed: %s", exc)
            return {"intent": "other", "confidence": 0.0, "model": "none"}
