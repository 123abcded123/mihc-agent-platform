"""
LangGraph 图状态定义（对齐《项目文档》5.5 多 Agent 编排）

一次请求的状态包在图节点间传递：
入口 → 输入安检 → Redis 上下文 → BERT 意图 → 任务规划
     → 步骤执行(循环，动态路由到专业 Agent) → 结果合并 → 引用检查 → 输出护栏 → 输出
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Annotated, TypedDict


def merge_lists(left: list, right: list) -> list:
    """LangGraph 归并函数：列表字段累加。"""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    # ---- 请求标识 ----
    query: str                                # 用户原始问题
    session_id: str                           # 会话 ID
    tenant_id: str                            # 租户
    user_id: str                              # 用户
    trace_id: str                             # 链路 ID

    # ---- 上下文 ----
    history: List[Dict[str, str]]             # Redis 中的最近消息

    # ---- 安检 ----
    guard_blocked: bool                       # 输入被拦截
    guard_reason: str                         # 拦截原因

    # ---- 意图与规划 ----
    intent: str                               # 意图标签（如 literature_search）
    intent_confidence: float                  # 意图置信度
    intent_model: str                         # bert / llm_fallback
    plan: List[Dict[str, Any]]                # 任务规划步骤列表
    step_index: int                           # 当前执行步骤下标
    max_steps: int                            # 最大步数（防死循环）

    # ---- 执行产物 ----
    agent_outputs: Dict[str, Any]             # 各 Agent 的中间产物（上下文隔离）
    agent_chain: Annotated[List[str], merge_lists]   # 已执行 Agent 链
    agent_traces: Annotated[List[Dict[str, Any]], merge_lists]  # 执行轨迹（延迟/状态）

    # ---- 最终输出 ----
    merged_output: str                        # 合并后的文本
    answer: str                               # 最终答案
    citations: List[Dict[str, Any]]           # 引用列表
    risks: List[Dict[str, str]]               # 风险提示
    output_blocked: bool                      # 输出被护栏拦截
