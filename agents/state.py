
from __future__ import annotations

from typing import List, Dict, Any, Optional, Annotated, TypedDict

def merge_lists(left: list, right: list) -> list:
    return (left or []) + (right or [])

class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    tenant_id: str
    user_id: str
    trace_id: str

    history: List[Dict[str, str]]

    guard_blocked: bool
    guard_reason: str

    intent: str
    intent_confidence: float
    intent_model: str
    plan: List[Dict[str, Any]]
    step_index: int
    max_steps: int

    agent_outputs: Dict[str, Any]
    agent_chain: Annotated[List[str], merge_lists]
    agent_traces: Annotated[List[Dict[str, Any]], merge_lists]

    merged_output: str
    answer: str
    citations: List[Dict[str, Any]]
    risks: List[Dict[str, str]]
    output_blocked: bool
