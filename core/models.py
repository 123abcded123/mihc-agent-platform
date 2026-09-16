
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

class Citation(BaseModel):
    chunk_id: str = Field(description="片段 ID")
    title: str = Field(default="", description="章节/文档标题")
    source: str = Field(default="", description="原始文件或 DOI")
    text_snippet: str = Field(default="", description="证据片段摘录")
    score: float = Field(default=0.0, description="重排后相关性得分")

class Risk(BaseModel):
    level: str = Field(default="info", description="info / warning / critical")
    message: str = Field(description="风险描述")

class IntentResult(BaseModel):
    intent: str = Field(description="意图标签，如 literature_search")
    confidence: float = Field(description="softmax 置信度 0~1")
    model: str = Field(default="bert", description="识别模型名（bert / llm 回退）")

class PlanStep(BaseModel):
    step_id: int
    description: str
    agent: str = Field(description="负责该步骤的 Agent 名")
    depends_on: List[int] = Field(default_factory=list, description="依赖步骤 ID")
    check: str = Field(default="", description="该步骤的验证方式")

class ChatRequest(BaseModel):
    query: str = ""
    question: Optional[str] = None
    session_id: Optional[str] = None
    tenant_id: str = "mihc"
    user_id: str = "anonymous"
    project_id: Optional[str] = None
    file_ids: List[str] = Field(default_factory=list)

    def effective_query(self) -> str:
        return (self.question or self.query).strip()

class AgentTrace(BaseModel):
    agent: str
    intent: str = ""
    status: str = "ok"
    latency_ms: int = 0
    detail: str = ""

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: List[Citation] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)
    trace_id: str
    agent_chain: List[str] = Field(default_factory=list)
    agent_traces: List[AgentTrace] = Field(default_factory=list)
    branch: str = "general"
    workflow_status: str = "completed"
    task_id: Optional[str] = None
    next_action: str = ""
    classification: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

class IngestResult(BaseModel):
    doc_id: str
    file_name: str
    chunks: int
    status: str = "ok"
    message: str = ""

class ProjectCreateRequest(BaseModel):
    project_name: str = Field(min_length=1, max_length=200)
    description: str = ""
    tenant_id: str = "mihc"
    user_id: str = "anonymous"
    member_ids: List[str] = Field(default_factory=list)

class SampleCreateRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=128)
    group: str = ""
    batch: str = ""
    sample_type: str = ""
    tenant_id: str = "mihc"
    user_id: str = "anonymous"

class MarkerCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    channel: str = ""
    antibody: str = ""
    threshold: Optional[float] = None
    unit: str = ""
    tenant_id: str = "mihc"
    user_id: str = "anonymous"

class MihcAnalyzeRequest(BaseModel):
    project_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    file_ids: List[str] = Field(default_factory=list)
    tenant_id: str = "mihc"
    user_id: str = "anonymous"
    depends_on_run_ids: List[str] = Field(default_factory=list)
    template_version: str = "mihc_default_v1"

class LiteratureIngestRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300, description="PubMed 检索词，如 mIHC tumor microenvironment")
    max_results: int = Field(default=10, ge=1, le=50, description="检索结果数上限")
    max_download: int = Field(default=5, ge=1, le=20, description="下载入库的开放获取文献数上限")

class WorkflowResponse(BaseModel):
    status: str
    branch: str
    task_id: Optional[str] = None
    analysis_run_id: Optional[str] = None
    input_refs: List[str] = Field(default_factory=list)
    output_refs: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    next_action: str = ""
    classification: Optional[Dict[str, Any]] = None

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class UserInfo(BaseModel):
    username: str
    tenant_id: str
    role: str
    display_name: str

class AuthResponse(BaseModel):
    token: str
    user: UserInfo

class EvalReport(BaseModel):
    hit_rate_at_k: dict = Field(default_factory=dict)
    recall_at_k: dict = Field(default_factory=dict)
    mrr: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    citation_correctness: float = 0.0
    safety_rejection_rate: float = 0.0
    latency_avg_ms: float = 0.0
    total_samples: int = 0
