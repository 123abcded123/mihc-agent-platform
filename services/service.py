"""
平台服务层：把存储、检索、安全、Agent 图组装成完整业务能力

- chat：完整对话链路（对齐《项目文档》5.2 端到端流程）；
- ingest：文档入库；
- 会话管理：Redis 读写（多会话/多用户隔离）；
- 审计：PostgreSQL audit_logs + badcase 留存；
- 观测：Prometheus 指标 + trace_id。
"""

from __future__ import annotations

import logging
import time
import threading
import os
import uuid
from typing import Dict, Any, Optional

from config import Config
from core.errors import MIHCError, SessionNotFoundError, PermissionDeniedError
from core.models import ChatRequest, ChatResponse, IngestResult
from infra.postgres import Database
from infra.redis_store import SessionStore
from infra.mongo_store import MongoStore
from agents.graph import AgentGraph
from rag.ingestion.pipeline import IngestionPipeline
from observability import metrics as obs_metrics
from observability.tracing import new_trace_id, span
from mihc.intent import MihcIntentClassifier
from mihc.routing import choose_branch, clarification_for
from mihc.object_store import LocalObjectStore
from mihc.table_analysis import analyze_tables, load_table
from services.auth import get_token_service, bootstrap_admin

logger = logging.getLogger(__name__)


class MIHCPlatform:
    """MIHC 医疗科研智能 Agent 平台服务。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.db = Database(self.config.postgres.url)
        self.db.init_tables()
        self.sessions = SessionStore(self.config)
        self.mongo = MongoStore(self.config)
        self.graph_holder = AgentGraph(self.config)
        self.graph = self.graph_holder.build()
        self.mihc_intent = MihcIntentClassifier(self.config.mihc.intent_min_confidence)
        self.object_store = LocalObjectStore(self.config.mihc.object_storage_dir)
        self.tokens = get_token_service(self.config)
        bootstrap_admin(self.db, self.config)

        # 后台预热：首次调用时在后台线程加载嵌入/重排模型（不阻塞启动）
        warmup_thread = threading.Thread(target=self.graph_holder.warmup, daemon=True, name="model-warmup")
        warmup_thread.start()
        logger.info("MIHC Platform initialized (tenant=%s)", self.config.platform.tenant_id)

    # ---- 文档入库 ----
    def ingest_document(self, file_path: str, *, version: str = "", source: str = "",
                        permission: str = "research_team") -> IngestResult:
        """文档入库：解析→切分→嵌入→Milvus+关键词双写→PG 元数据。"""
        with span("ingest.document", {"file": file_path}):
            embedder = self.graph_holder.embedder
            pipeline = IngestionPipeline(
                self.config, embedder,
                self.graph_holder.retriever.milvus,
                self.graph_holder.retriever.keyword,
                self.db,
            )
            result = pipeline.ingest(
                file_path, version=version, source=source,
                permission=permission, tenant_id=self.config.platform.tenant_id,
            )
            return IngestResult(**result)

    # ---- 对话 ----
    def chat(self, request: ChatRequest) -> ChatResponse:
        """完整对话链路（图执行 + 会话 + 审计 + 指标）。"""
        start = time.time()
        query = request.effective_query()
        if not query:
            raise MIHCError("问题不能为空", code="invalid_request", status_code=422)
        session_id = request.session_id
        # 会话创建/校验（多会话切换 + 用户隔离）
        if session_id and not self.sessions.exists(session_id):
            raise SessionNotFoundError(f"会话不存在: {session_id}")
        if not session_id:
            session_id = self.sessions.create_session(tenant_id=request.tenant_id, user_id=request.user_id)
        classification = self.mihc_intent.classify(query, has_files=bool(request.file_ids))
        trace_id = new_trace_id()
        branch = choose_branch(classification, project_id=request.project_id,
                               file_ids=request.file_ids,
                               min_confidence=self.config.mihc.intent_min_confidence)

        # 对话关联项目编号：登录用户校验项目权限，匿名用户仅做标注（不阻断）
        project_context = ""
        if request.project_id:
            project = self.db.get_project(request.project_id, request.tenant_id)
            if project and (request.user_id in project["member_ids"] or request.user_id == "anonymous"):
                project_context = f"当前客户项目：{request.project_id}（{project['project_name']}）"
            elif project and request.user_id != "anonymous":
                raise PermissionDeniedError("用户不是该项目的成员")
            else:
                project_context = f"当前客户项目：{request.project_id}（无权限或不存在，仅作编号标注）"

        if branch not in {"general", "product_consult", "experiment_design", "literature_recommendation"}:
            answer = clarification_for(branch, classification)
            self.sessions.append_message(session_id, "user", query,
                                         max_history=self.config.platform.max_history)
            self.sessions.append_message(session_id, "assistant", answer,
                                         max_history=self.config.platform.max_history)
            self.db.add_audit(trace_id=trace_id, session_id=session_id,
                              tenant_id=request.tenant_id, user_id=request.user_id,
                              query=query[:2000], intent=classification["intent"],
                              agent_chain="mihc_router", latency_ms=int((time.time() - start) * 1000),
                              status="needs_input", safety_event=branch)
            return ChatResponse(answer=answer, session_id=session_id,
                                risks=[{"level": "warning", "message": "当前任务尚未进入计算或报告生成阶段"}],
                                trace_id=trace_id, agent_chain=["mihc_router"], branch=branch,
                                workflow_status="clarifying" if branch.startswith("clarify") else branch,
                                next_action=branch, classification=classification)
        history = self.sessions.history(session_id)

        state = {
            "query": query,
            "session_id": session_id,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "trace_id": trace_id,
            "history": history,
            "max_steps": self.config.platform.max_agent_steps,
            "project_id": request.project_id or "",
            "project_context": project_context,
        }

        status = "ok"
        intent = "unknown"
        try:
            with span("chat.request", {"trace_id": trace_id, "session_id": session_id}):
                final_state = self.graph.invoke(state)
        except MIHCError as exc:
            status = "error"
            obs_metrics.record_request(intent, status, time.time() - start)
            raise
        except Exception as exc:  # noqa: BLE001
            status = "error"
            logger.exception("Chat pipeline failed: %s", exc)
            obs_metrics.record_request(intent, status, time.time() - start)
            raise MIHCError(f"处理失败: {exc}")

        intent = final_state.get("intent", "unknown")
        answer = final_state.get("answer", final_state.get("merged_output", ""))
        guard_blocked = final_state.get("guard_blocked", False)

        # 会话写回（用户消息 + 助手回复）
        self.sessions.append_message(session_id, "user", query,
                                     max_history=self.config.platform.max_history)
        self.sessions.append_message(session_id, "assistant", answer,
                                     max_history=self.config.platform.max_history)
        self.sessions.set_intent(session_id, intent)

        # 审计日志（对齐文档：没有日志就无法知道错误发生在哪一阶段）
        latency_ms = int((time.time() - start) * 1000)
        self.db.add_audit(
            trace_id=trace_id, session_id=session_id,
            tenant_id=request.tenant_id, user_id=request.user_id,
            query=query[:2000], intent=intent,
            agent_chain=", ".join(final_state.get("agent_chain", [])),
            latency_ms=latency_ms, status=status,
            safety_event=final_state.get("guard_reason", ""),
        )
        if status == "error" or guard_blocked:
            self.mongo.add_badcase(
                query=query, answer=answer, intent=intent, trace_id=trace_id,
                feedback="guard_blocked" if guard_blocked else "error",
                root_cause="guard" if guard_blocked else "pipeline_error",
            )

        obs_metrics.record_request(intent, status, time.time() - start)
        obs_metrics.LLM_TOKENS.labels(kind="completion").inc(obs_metrics.estimate_tokens(answer))

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            citations=[c for c in final_state.get("citations", [])][:10],
            risks=final_state.get("risks", []),
            trace_id=trace_id,
            agent_chain=final_state.get("agent_chain", []),
            agent_traces=final_state.get("agent_traces", []),
            branch=branch,
            classification=classification,
        )

    # ---- mIHC project and file operations ----
    def require_project(self, project_id: str, tenant_id: str, user_id: str) -> dict:
        project = self.db.get_project(project_id, tenant_id)
        if not project:
            raise MIHCError("项目不存在", code="project_not_found", status_code=404)
        if user_id not in project["member_ids"]:
            raise PermissionDeniedError("用户不是项目成员")
        return project

    def create_project(self, request) -> dict:
        project_id = f"P-{uuid.uuid4().hex[:12]}"
        return self.db.add_project(project_id=project_id, project_name=request.project_name,
                                   description=request.description, tenant_id=request.tenant_id,
                                   owner_id=request.user_id, member_ids=request.member_ids)

    def upload_project_file(self, project_id: str, tenant_id: str, user_id: str,
                            file_name: str, content_type: str, content: bytes, kind: str = "unknown") -> dict:
        self.require_project(project_id, tenant_id, user_id)
        if len(self.db.list_project_files(project_id, tenant_id)) >= self.config.mihc.max_project_files:
            raise MIHCError("项目文件数量超过上限", code="file_limit_exceeded", status_code=422)
        file_id = f"F-{uuid.uuid4().hex[:16]}"
        stored = self.object_store.put_bytes(project_id, file_id, file_name, content)
        suffix = os.path.splitext(file_name or "")[1].lower()
        inferred_kind = kind if kind != "unknown" else (
            "cell_table" if suffix in {".csv", ".tsv", ".xls", ".xlsx"} else
            "image" if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} else "unknown")
        return self.db.add_project_file(file_id=file_id, project_id=project_id, tenant_id=tenant_id,
                                        file_name=file_name, object_key=stored["object_key"],
                                        content_type=content_type or "application/octet-stream",
                                        extension=suffix, kind=inferred_kind,
                                        size_bytes=stored["size_bytes"], sha256=stored["sha256"],
                                        created_by=user_id)

    def analyze_project_tables(self, request) -> dict:
        self.require_project(request.project_id, request.tenant_id, request.user_id)
        if not request.file_ids:
            raise MIHCError("表格分析至少需要一个文件", code="missing_files", status_code=422)
        file_rows = [self.db.get_project_file(file_id, request.tenant_id) for file_id in request.file_ids]
        if any(not row or row["project_id"] != request.project_id for row in file_rows):
            raise PermissionDeniedError("文件不存在或不属于当前项目")
        run_id = f"AR-{uuid.uuid4().hex[:16]}"
        classification = self.mihc_intent.classify(request.question, has_files=True)
        self.db.create_analysis_run(run_id=run_id, project_id=request.project_id,
                                    tenant_id=request.tenant_id, user_id=request.user_id,
                                    branch="table_analysis", status="running",
                                    current_node="file_validation", input_refs=request.file_ids,
                                    config={"question": request.question, "template_version": request.template_version},
                                    model_version="deterministic-pandas-v1")
        try:
            samples = self.db.list_samples(request.project_id, request.tenant_id)
            markers = self.db.list_markers(request.project_id, request.tenant_id)
            cell_row = next((row for row in file_rows if row["kind"] == "cell_table"), None)
            omics_row = next((row for row in file_rows if row["kind"] == "omics_table"), None)
            if not cell_row:
                raise MIHCError("未找到细胞表格文件，请上传 CSV/Excel 并标记为 cell_table",
                                code="missing_cell_table", status_code=422)
            cell_df = load_table(str(self.object_store._path(cell_row["object_key"])),
                                 self.config.mihc.max_table_rows)
            omics_df = load_table(str(self.object_store._path(omics_row["object_key"])),
                                  self.config.mihc.max_table_rows) if omics_row else None
            result = analyze_tables(cell_df, omics_df, samples, markers,
                                    classification["entities"].get("markers"), request.question)
            status = "completed" if result["status"] == "passed" else "data_invalid"
            next_action = "generate_report" if status == "completed" else "clarify"
            output_refs = [f"analysis:{run_id}"]
            self.db.update_analysis_run(run_id, status=status, current_node="validated",
                                        result=result, output_refs=output_refs,
                                        errors=result.get("errors", []), next_action=next_action)
            return {"status": status, "branch": "table_analysis", "analysis_run_id": run_id,
                    "input_refs": request.file_ids, "output_refs": output_refs, "data": result,
                    "errors": result.get("errors", []), "next_action": next_action,
                    "classification": classification}
        except MIHCError:
            self.db.update_analysis_run(run_id, status="data_invalid", current_node="failed",
                                        errors=[{"code": "validation_failed", "message": "输入文件无法通过校验"}],
                                        next_action="clarify")
            raise
        except Exception as exc:
            logger.exception("mIHC table analysis failed: %s", exc)
            self.db.update_analysis_run(run_id, status="tool_failed", current_node="failed",
                                        errors=[{"code": "tool_failed", "message": str(exc)}], next_action="retry")
            raise MIHCError(f"表格分析失败: {exc}", code="analysis_failed", status_code=422)

    # ---- mIHC 文献下载入库 ----
    def ingest_literature(self, *, query: str, max_results: int = 10, max_download: int = 5,
                          tenant_id: str = "mihc") -> dict:
        """PubMed 检索 mIHC 文献 → 下载开放获取 PDF → 解析入库（Milvus+BM25+PG）。"""
        from pathlib import Path
        from mihc.literature import PubMedLiterature

        downloader = PubMedLiterature(self.config.mihc.literature_dir)
        files = downloader.download(query, max_results=max_results, max_download=max_download)
        if not files:
            return {"status": "completed", "query": query, "downloaded": 0, "ingested": 0,
                    "details": [], "message": "未检索到可下载的开放获取文献，请调整检索词"}
        embedder = self.graph_holder.embedder
        pipeline = IngestionPipeline(self.config, embedder, self.graph_holder.retriever.milvus,
                                     self.graph_holder.retriever.keyword, self.db)
        details = []
        for item in files:
            entry = {"file": Path(item["file_path"]).name, "source": item["source"]}
            try:
                result = pipeline.ingest(item["file_path"], source=item["source"],
                                         permission="research_team", tenant_id=tenant_id)
                entry.update({"status": "ingested", "doc_id": result["doc_id"], "chunks": result["chunks"]})
            except Exception as exc:  # noqa: BLE001
                logger.warning("文献入库失败 %s: %s", item["file_path"], exc)
                entry.update({"status": "failed", "error": str(exc)[:200]})
            details.append(entry)
        ingested = sum(1 for d in details if d.get("status") == "ingested")
        return {"status": "completed", "query": query, "downloaded": len(files),
                "ingested": ingested, "details": details}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        payload = self.sessions.get(session_id)
        if not payload:
            raise SessionNotFoundError(f"会话不存在: {session_id}")
        return payload

    def delete_session(self, session_id: str) -> None:
        if not self.sessions.exists(session_id):
            raise SessionNotFoundError(f"会话不存在: {session_id}")
        self.sessions.clear(session_id)
