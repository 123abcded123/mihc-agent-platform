
from __future__ import annotations

import re
from typing import Any, Dict, List

MIHC_INTENTS = {
    "product_consult",
    "experiment_design",
    "table_analysis",
    "image_analysis",
    "literature_recommendation",
    "report_generation",
    "other",
}

INTENT_TO_AGENT = {
    "product_consult": "knowledge_agent",
    "experiment_design": "experiment_agent",
    "table_analysis": "data_analysis_agent",
    "image_analysis": "human_review",
    "literature_recommendation": "literature_agent",
    "report_generation": "report_agent",
}

KNOWN_MARKERS = (
    "CD3", "CD4", "CD8", "FOXP3", "PD-L1", "PDL1", "IFNG", "IFN-γ",
    "Ki-67", "KI67", "CD68", "CD163", "PanCK", "CK", "MHC-I", "MHCII",
)

_INTENT_PATTERNS = {
    "product_consult": (
        "hyperview", "产品", "平台", "panel", "抗体库", "样本接收", "适合什么样本",
        "服务能力", "报价", "交期", "石蜡切片",
    ),
    "experiment_design": (
        "预实验", "实验设计", "实验方案", "怎么做", "如何做", "panel", "marker",
        "对照组", "重复数", "样本要求",
    ),
    "table_analysis": (
        "表格", "csv", "excel", "细胞表", "阳性率", "比较", "统计", "数据分析",
        "治疗组", "对照组", "转录组结果",
    ),
    "image_analysis": (
        "原始图像", "切片图", "图像分析", "roi", "空间分析", "细胞分割", "像素",
        "荧光图", "组织图像",
    ),
    "literature_recommendation": (
        "文献", "论文", "推荐几篇", "研究进展", "sop", "依据", "pubmed", "doi",
    ),
    "report_generation": (
        "生成报告", "分析报告", "导出报告", "汇总结果", "报告", "pdf", "html",
    ),
}

_REQUIRED_INPUTS = {
    "product_consult": [],
    "experiment_design": ["research_purpose", "sample_type"],
    "table_analysis": ["cell_table"],
    "image_analysis": ["image_or_cell_result"],
    "literature_recommendation": [],
    "report_generation": ["completed_analysis_run"],
    "other": [],
}

def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)

def extract_entities(query: str) -> Dict[str, Any]:
    markers: List[str] = []
    for marker in KNOWN_MARKERS:
        if marker.lower() in query.lower() and marker not in markers:
            markers.append(marker)

    groups: List[str] = []
    group_aliases = {
        "treatment": ("treatment", "治疗组", "处理组", "实验组"),
        "control": ("control", "对照组", "对照", "vehicle", "placebo"),
    }
    for normalized, aliases in group_aliases.items():
        if _contains_any(query, aliases):
            groups.append(normalized)

    analysis: List[str] = []
    analysis_terms = {
        "positive_rate": ("阳性率", "阳性比例", "positive rate"),
        "group_comparison": ("比较", "差异", "组间", "t检验", "显著性"),
        "correlation": ("相关", "关联", "correlation"),
        "spatial_distance": ("距离", "邻近", "空间分析"),
    }
    for name, terms in analysis_terms.items():
        if _contains_any(query, terms):
            analysis.append(name)

    files = re.findall(r"[\w.-]+\.(?:csv|xlsx?|tsv|tif{1,2}|png|jpe?g)", query, flags=re.I)
    return {"markers": markers, "groups": groups, "analysis": analysis, "file_names": files}

class MihcIntentClassifier:

    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence

    def classify(self, query: str, *, has_files: bool = False) -> Dict[str, Any]:
        query = (query or "").strip()
        entities = extract_entities(query)
        matched = [name for name, terms in _INTENT_PATTERNS.items() if _contains_any(query, terms)]

        if has_files and ("table_analysis" in matched or any(ext in query.lower() for ext in (".csv", ".xlsx", ".xls", ".tsv"))):
            matched = ["table_analysis"] + [item for item in matched if item != "table_analysis"]

        if not matched:
            return {
                "intent": "other",
                "intents": ["other"],
                "confidence": 0.20,
                "model": "mihc_rule",
                "entities": entities,
                "required_inputs": [],
                "risk_level": "low",
            }

        intents = list(dict.fromkeys(matched))
        primary = intents[0]
        confidence = 0.93 if len(intents) == 1 else 0.86
        if primary == "table_analysis" and has_files:
            confidence = 0.97
        risk_level = "medium" if primary in {"experiment_design", "table_analysis", "image_analysis"} else "low"
        if _contains_any(query, ("患者身份", "身份证", "病历号", "诊断", "处方")):
            risk_level = "high"
        required = list(dict.fromkeys(item for intent in intents for item in _REQUIRED_INPUTS[intent]))
        return {
            "intent": primary,
            "intents": intents,
            "confidence": confidence,
            "model": "mihc_rule",
            "entities": entities,
            "required_inputs": required,
            "risk_level": risk_level,
        }
