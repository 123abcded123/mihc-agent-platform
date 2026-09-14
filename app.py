"""
app.py —— MIHC 医疗科研智能 Agent 平台 Web 服务入口

对齐《项目文档》5.2 端到端流程：
  Vue3 对话页面 → FastAPI 接口 → 身份/权限/敏感信息检查 → Redis 会话
  → BERT 意图识别 → LangGraph 动态路由 → 检索(ES+Milvus+RRF+Reranker)
  → 多 Agent 编排 → Qwen3-32B 生成 → DFA+Guardrail 安全检查
  → Langfuse/OTel 记录 → 返回答案、引用、风险和 trace_id

HTTP 接口：
  POST /api/v1/chat                 对话主入口
  GET  /api/v1/sessions/{sid}       会话历史
  DELETE /api/v1/sessions/{sid}     清空会话
  POST /api/v1/documents/ingest     文档入库
  GET  /api/v1/documents            已入库文档列表
  GET  /api/v1/badcases             badcase 列表（闭环管理）
  GET  /api/v1/eval/results         评测结果
  GET  /health                      探活
  GET  /metrics                     Prometheus 指标
"""

from __future__ import annotations

import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config import Config
from core.errors import MIHCError
from core.models import (ChatRequest, ChatResponse, IngestResult, ProjectCreateRequest,
                         SampleCreateRequest, MarkerCreateRequest, MihcAnalyzeRequest,
                         WorkflowResponse, RegisterRequest, LoginRequest, AuthResponse, UserInfo)
from services.service import MIHCPlatform
from services.auth import hash_password, verify_password
from observability.otel import init_otel
from observability.tracing import init_langfuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---- 初始化 ----
config = Config()
init_otel(config)
init_langfuse(config)

app = FastAPI(
    title="MIHC Agent Platform",
    description="医疗科研领域智能 Agent 平台（文献检索/知识问答/检验解读/数据分析/实验设计）",
    version="1.0.0",
)

# 跨域：前端与后端在 Sealos 上是两个独立公网域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=False,   # 使用 Bearer token 鉴权，不依赖 Cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

platform_service = MIHCPlatform(config)


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """依赖注入：校验 Bearer token，返回 {sub, tenant_id, role}。"""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, detail={"error": {"code": "unauthorized", "message": "请先登录"}})
    payload = platform_service.tokens.verify(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(401, detail={"error": {"code": "token_invalid", "message": "登录已过期，请重新登录"}})
    return payload


# ---- 异常处理 ----
@app.exception_handler(MIHCError)
async def mihc_error_handler(request: Request, exc: MIHCError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": str(exc)}})


@app.exception_handler(413)
async def request_too_large(request: Request, exc):
    return JSONResponse(status_code=413, content={
        "error": {"code": "file_too_large",
                  "message": f"文件过大，最大允许 {config.api.max_upload_size_mb}MB"}})


# ---- 对话 ----
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    """对话主入口：返回答案、引用、风险与 trace_id。

    未登录：仅开放公开产品咨询与通用问答；
    已登录：请求绑定 token 中的用户与租户（携带 Bearer token 即可）。
    """
    if authorization and authorization.lower().startswith("bearer "):
        payload = platform_service.tokens.verify(authorization.split(" ", 1)[1].strip())
        if payload:
            request.user_id = payload["sub"]
            request.tenant_id = payload["tenant_id"]
    try:
        return platform_service.chat(request)
    except MIHCError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict())


# ---- 登录鉴权 ----
@app.post("/api/v1/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    if not config.auth.allow_register:
        raise HTTPException(403, detail={"error": {"code": "register_disabled", "message": "注册已关闭"}})
    if platform_service.db.get_user(request.username):
        raise HTTPException(409, detail={"error": {"code": "username_taken", "message": "用户名已存在"}})
    user = platform_service.db.create_user(username=request.username,
                                           password_hash=hash_password(request.password),
                                           tenant_id=config.platform.tenant_id,
                                           display_name=request.display_name)
    token = platform_service.tokens.issue(user)
    return {"token": token, "user": UserInfo(**user).model_dump()}


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    user = platform_service.db.get_user(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(401, detail={"error": {"code": "bad_credentials", "message": "用户名或密码错误"}})
    token = platform_service.tokens.issue(user)
    return {"token": token, "user": UserInfo(**user).model_dump()}


@app.get("/api/v1/auth/me", response_model=UserInfo)
async def me(user: dict = Depends(get_current_user)):
    record = platform_service.db.get_user(user["sub"])
    if not record:
        raise HTTPException(401, detail={"error": {"code": "user_gone", "message": "用户不存在"}})
    return UserInfo(**record).model_dump()


# ---- 客户 mIHC 项目工作台（全部需要登录）----
@app.post("/api/v1/projects")
async def create_project(request: ProjectCreateRequest, user: dict = Depends(get_current_user)):
    request.tenant_id = user["tenant_id"]
    request.user_id = user["sub"]
    return platform_service.create_project(request)


@app.get("/api/v1/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    return {"projects": platform_service.db.list_projects(user["tenant_id"], user["sub"])}


@app.get("/api/v1/projects/{project_id}")
async def get_project(project_id: str, user: dict = Depends(get_current_user)):
    platform_service.require_project(project_id, user["tenant_id"], user["sub"])
    return {"project": platform_service.db.get_project(project_id, user["tenant_id"]),
            "samples": platform_service.db.list_samples(project_id, user["tenant_id"]),
            "markers": platform_service.db.list_markers(project_id, user["tenant_id"]),
            "files": platform_service.db.list_project_files(project_id, user["tenant_id"])}


@app.post("/api/v1/projects/{project_id}/samples")
async def add_sample(project_id: str, request: SampleCreateRequest, user: dict = Depends(get_current_user)):
    platform_service.require_project(project_id, user["tenant_id"], user["sub"])
    platform_service.db.add_samples([{"sample_id": request.sample_id, "project_id": project_id,
                                      "group": request.group, "batch": request.batch,
                                      "sample_type": request.sample_type, "tenant_id": user["tenant_id"]}])
    return {"status": "created", "sample_id": request.sample_id}


@app.post("/api/v1/projects/{project_id}/markers")
async def add_marker(project_id: str, request: MarkerCreateRequest, user: dict = Depends(get_current_user)):
    platform_service.require_project(project_id, user["tenant_id"], user["sub"])
    import uuid
    return platform_service.db.add_marker(marker_id=f"M-{uuid.uuid4().hex[:12]}", project_id=project_id,
                                           name=request.name, channel=request.channel,
                                           antibody=request.antibody, threshold=request.threshold,
                                           unit=request.unit, tenant_id=user["tenant_id"])


@app.post("/api/v1/projects/{project_id}/files")
async def upload_project_file(project_id: str, file: UploadFile = File(...),
                              kind: str = Form("unknown"), user: dict = Depends(get_current_user)):
    content = await file.read()
    if len(content) > config.api.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, detail={"error": {"code": "file_too_large", "message": "文件过大"}})
    return platform_service.upload_project_file(project_id, user["tenant_id"], user["sub"],
                                                file.filename or "file", file.content_type or "", content, kind)


@app.post("/api/v1/projects/{project_id}/analysis/table", response_model=WorkflowResponse)
async def analyze_project_table(project_id: str, request: MihcAnalyzeRequest,
                                user: dict = Depends(get_current_user)):
    if request.project_id != project_id:
        raise HTTPException(422, detail={"error": {"code": "project_mismatch", "message": "路径和请求体项目不一致"}})
    request.tenant_id = user["tenant_id"]
    request.user_id = user["sub"]
    return platform_service.analyze_project_tables(request)


@app.get("/api/v1/analysis-runs/{run_id}")
async def get_analysis_run(run_id: str, user: dict = Depends(get_current_user)):
    run = platform_service.db.get_analysis_run(run_id, user["tenant_id"])
    if not run:
        raise HTTPException(404, detail={"error": {"code": "run_not_found", "message": "分析任务不存在"}})
    return run


# ---- 会话 ----
@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        return platform_service.get_session(session_id)
    except MIHCError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict())


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        platform_service.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except MIHCError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict())


# ---- 文档管理 ----
@app.post("/api/v1/documents/ingest", response_model=IngestResult)
async def ingest_document(file: UploadFile = File(...),
                          version: str = Form(""),
                          source: str = Form(""),
                          permission: str = Form("research_team"),
                          user: dict = Depends(get_current_user)):
    """文档入库（PDF/DOCX/MD/TXT）→ Milvus + 关键词库 + PG 元数据。"""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".pdf", ".docx", ".md", ".txt"):
        raise HTTPException(422, detail={"error": {"code": "unsupported_file", "message": f"不支持的类型: {suffix}"}})

    size = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as tmp:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.api.max_upload_size_mb * 1024 * 1024:
                    raise HTTPException(413, detail={"error": {"code": "file_too_large", "message": "文件过大"}})
                tmp.write(chunk)
            tmp_path = tmp.name
        return platform_service.ingest_document(tmp_path, version=version, source=source, permission=permission)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/v1/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    return {"documents": platform_service.db.list_documents(config.platform.tenant_id)}


# ---- badcase 闭环 ----
@app.get("/api/v1/badcases")
async def list_badcases(stage: Optional[str] = None, limit: int = 100,
                        user: dict = Depends(get_current_user)):
    return {"badcases": platform_service.mongo.list_badcases(stage=stage, limit=limit)}


# ---- 评测 ----
@app.get("/api/v1/eval/results")
async def eval_results(limit: int = 20, user: dict = Depends(get_current_user)):
    return {"results": platform_service.mongo.list_eval_results(limit=limit)}


# ---- 运维 ----
@app.get("/health")
async def health():
    return {"status": "healthy", "service": config.observability.otel_service_name}


@app.get("/metrics")
async def metrics():
    if not config.observability.prometheus_enabled:
        return JSONResponse(status_code=404, content={"error": "metrics disabled"})
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---- 静态挂载（必须放在所有 API 路由之后，避免吞掉 /health /metrics 等路由）----
# /data 必须先于根路径挂载，否则会被前端 StaticFiles 的 "/" 路由吞掉。
os.makedirs("./data", exist_ok=True)
app.mount("/data", StaticFiles(directory="./data"), name="data")

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # 以配置中的 host/port 启动服务
    uvicorn.run(app, host=config.api.host, port=config.api.port)
