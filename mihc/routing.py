"""mIHC 工作流分支选择。

返回的是受控节点名，不允许模型直接拼接 LangGraph 节点名。
"""

from __future__ import annotations

from typing import Any, Dict

from .intent import INTENT_TO_AGENT


def choose_branch(classification: Dict[str, Any], *, project_id: str | None,
                  file_ids: list[str], completed_tasks: bool = False,
                  min_confidence: float = 0.75) -> str:
    if classification.get("risk_level") == "high":
        return "human_review"
    if classification.get("confidence", 0.0) < min_confidence:
        return "clarify"

    intent = classification.get("intent", "other")
    if intent == "table_analysis" and not file_ids:
        return "clarify_missing_file"
    if intent == "image_analysis" and not file_ids:
        return "clarify_missing_image"
    if intent == "report_generation" and not completed_tasks:
        return "run_prerequisites"
    if intent == "experiment_design" and not project_id:
        # 公共知识可以回答实验原则，但不能生成客户项目方案。
        return "clarify_missing_project"
    if intent == "other":
        return "general"
    return intent


def route_agent(classification: Dict[str, Any]) -> str:
    """把业务分支映射到已有 Agent；报告/图像由领域服务或人工审核处理。"""
    return INTENT_TO_AGENT.get(classification.get("intent", "other"), "knowledge_agent")


def clarification_for(branch: str, classification: Dict[str, Any]) -> str:
    prompts = {
        "clarify": "为了避免把问题路由到错误的科研流程，请补充您要处理的对象：产品资料、实验设计、表格统计、图像分析、文献推荐或报告生成？",
        "clarify_missing_file": "请先上传当前项目的细胞表格或组学表格，并确认 sample_id、group 和 marker/feature 字段的含义。",
        "clarify_missing_image": "请上传原始切片/ROI 或已通过质控的细胞级结果，并补充通道映射、像素尺寸和模型版本。",
        "clarify_missing_project": "实验方案需要绑定客户项目。请先创建或选择项目，并提供样本类型、研究目的和候选 marker。",
        "run_prerequisites": "报告生成需要已通过质量检查的 analysis_run_id。请先完成数据校验和统计分析，再提交报告任务。",
        "human_review": "该请求涉及高风险内容或需要业务承诺，系统已停止自动处理并建议转人工审核。",
    }
    return prompts.get(branch, "当前请求需要补充更多信息后才能继续。")
