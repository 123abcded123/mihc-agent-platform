
from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    doc_id = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    file_name = Column(String(512), nullable=False)
    source = Column(String(512), default="")
    version = Column(String(32), default="")
    tenant_id = Column(String(64), default="mihc", index=True)
    permission = Column(String(256), default="research_team")
    embedding_model = Column(String(64), default="BGE-M3")
    status = Column(String(32), default="ingested")
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = Column(String(64), primary_key=True)
    doc_id = Column(String(64), ForeignKey("documents.doc_id"), index=True)
    title = Column(String(512), default="")
    text_snippet = Column(Text, default="")
    source = Column(String(512), default="")
    version = Column(String(32), default="")
    tenant_id = Column(String(64), default="mihc", index=True)
    embedding_model = Column(String(64), default="BGE-M3")
    created_at = Column(DateTime, default=datetime.now)

class Permission(Base):
    __tablename__ = "permissions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), index=True)
    user_id = Column(String(64), index=True)
    doc_id = Column(String(64), index=True)
    allow = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(64), index=True)
    session_id = Column(String(64), index=True)
    tenant_id = Column(String(64), index=True)
    user_id = Column(String(64))
    query = Column(Text)
    intent = Column(String(64))
    agent_chain = Column(String(512))
    latency_ms = Column(Integer, default=0)
    status = Column(String(32), default="ok")
    safety_event = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.now, index=True)

class Badcase(Base):
    __tablename__ = "badcases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(64), index=True)
    session_id = Column(String(64))
    query = Column(Text)
    answer = Column(Text)
    intent = Column(String(64))
    stage = Column(String(32), default="collected")
    feedback = Column(String(64), default="")
    expert_label = Column(String(64), default="")
    root_cause = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.now, index=True)

class Project(Base):

    __tablename__ = "projects"
    project_id = Column(String(64), primary_key=True)
    project_name = Column(String(200), nullable=False)
    description = Column(Text, default="")
    tenant_id = Column(String(64), default="mihc", index=True)
    owner_id = Column(String(64), default="anonymous", index=True)
    member_ids = Column(Text, default="[]")
    status = Column(String(32), default="active")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Sample(Base):
    __tablename__ = "samples"
    sample_id = Column(String(128), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.project_id"), index=True)
    group = Column(String(128), default="")
    batch = Column(String(128), default="")
    sample_type = Column(String(128), default="")
    tenant_id = Column(String(64), default="mihc", index=True)
    created_at = Column(DateTime, default=datetime.now)

class Marker(Base):
    __tablename__ = "markers"
    marker_id = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    project_id = Column(String(64), ForeignKey("projects.project_id"), index=True)
    name = Column(String(128), nullable=False)
    channel = Column(String(128), default="")
    antibody = Column(String(256), default="")
    threshold = Column(Float, nullable=True)
    unit = Column(String(128), default="")
    tenant_id = Column(String(64), default="mihc", index=True)
    created_at = Column(DateTime, default=datetime.now)

class ProjectFile(Base):
    __tablename__ = "project_files"
    file_id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.project_id"), index=True)
    tenant_id = Column(String(64), default="mihc", index=True)
    file_name = Column(String(512), nullable=False)
    object_key = Column(String(1024), nullable=False)
    content_type = Column(String(128), default="application/octet-stream")
    extension = Column(String(16), default="")
    kind = Column(String(64), default="unknown")
    size_bytes = Column(Integer, default=0)
    sha256 = Column(String(64), default="")
    status = Column(String(32), default="uploaded")
    created_by = Column(String(64), default="anonymous")
    created_at = Column(DateTime, default=datetime.now)

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    run_id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.project_id"), index=True)
    tenant_id = Column(String(64), default="mihc", index=True)
    user_id = Column(String(64), default="anonymous")
    branch = Column(String(64), default="")
    status = Column(String(32), default="received")
    current_node = Column(String(64), default="received")
    input_refs = Column(Text, default="[]")
    output_refs = Column(Text, default="[]")
    config_json = Column(Text, default="{}")
    result_json = Column(Text, default="{}")
    errors_json = Column(Text, default="[]")
    next_action = Column(String(128), default="")
    model_version = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Report(Base):
    __tablename__ = "reports"
    report_id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.project_id"), index=True)
    run_id = Column(String(64), index=True)
    tenant_id = Column(String(64), default="mihc", index=True)
    status = Column(String(32), default="draft")
    template_version = Column(String(64), default="mihc_default_v1")
    json_object_key = Column(String(1024), default="")
    html_object_key = Column(String(1024), default="")
    review_status = Column(String(32), default="pending")
    created_at = Column(DateTime, default=datetime.now)

class User(Base):

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    tenant_id = Column(String(64), default="mihc", index=True)
    role = Column(String(32), default="customer")
    display_name = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.now)

Index("ix_chunks_tenant_doc", Chunk.tenant_id, Chunk.doc_id)
Index("ix_project_files_project_kind", ProjectFile.project_id, ProjectFile.kind)
Index("ix_analysis_runs_project_status", AnalysisRun.project_id, AnalysisRun.status)

class Database:

    def __init__(self, url: str):
        if url.startswith("sqlite"):
            path = url.replace("sqlite:///", "")
            if path and path != ":memory:":
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.backend = "sqlite" if url.startswith("sqlite") else "postgresql"
        logger.info("Database backend: %s", self.backend)

    def init_tables(self):
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self.session_factory()

    def add_document(self, *, doc_id: str, file_name: str, source: str = "", version: str = "",
                     tenant_id: str = "mihc", permission: str = "research_team",
                     embedding_model: str = "BGE-M3", chunk_count: int = 0) -> None:
        with self.session() as s:
            s.add(Document(doc_id=doc_id, file_name=file_name, source=source, version=version,
                           tenant_id=tenant_id, permission=permission,
                           embedding_model=embedding_model, chunk_count=chunk_count))
            s.commit()

    def list_documents(self, tenant_id: str = "mihc") -> List[dict]:
        with self.session() as s:
            rows = s.query(Document).filter(Document.tenant_id == tenant_id).order_by(Document.created_at.desc()).all()
            return [
                {"doc_id": r.doc_id, "file_name": r.file_name, "source": r.source,
                 "version": r.version, "permission": r.permission,
                 "embedding_model": r.embedding_model, "chunk_count": r.chunk_count,
                 "created_at": r.created_at.isoformat() if r.created_at else ""}
                for r in rows
            ]

    @staticmethod
    def _json_load(raw: str, default):
        try:
            value = json.loads(raw or "")
            return value if value is not None else default
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _project_dict(row: Project) -> dict:
        return {
            "project_id": row.project_id,
            "project_name": row.project_name,
            "description": row.description or "",
            "tenant_id": row.tenant_id,
            "owner_id": row.owner_id,
            "member_ids": Database._json_load(row.member_ids, []),
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "updated_at": row.updated_at.isoformat() if row.updated_at else "",
        }

    def add_project(self, *, project_id: str, project_name: str, description: str = "",
                    tenant_id: str = "mihc", owner_id: str = "anonymous",
                    member_ids: Optional[List[str]] = None) -> dict:
        members = sorted(set((member_ids or []) + [owner_id]))
        with self.session() as s:
            s.add(Project(project_id=project_id, project_name=project_name, description=description,
                          tenant_id=tenant_id, owner_id=owner_id,
                          member_ids=json.dumps(members, ensure_ascii=False)))
            s.commit()
            return self._project_dict(s.get(Project, project_id))

    def get_project(self, project_id: str, tenant_id: Optional[str] = None) -> Optional[dict]:
        with self.session() as s:
            row = s.get(Project, project_id)
            if not row or (tenant_id and row.tenant_id != tenant_id):
                return None
            return self._project_dict(row)

    def list_projects(self, tenant_id: str = "mihc", user_id: Optional[str] = None) -> List[dict]:
        with self.session() as s:
            rows = s.query(Project).filter(Project.tenant_id == tenant_id).order_by(Project.created_at.desc()).all()
            result = []
            for row in rows:
                item = self._project_dict(row)
                if user_id and user_id not in item["member_ids"]:
                    continue
                result.append(item)
            return result

    def add_samples(self, rows: List[dict]) -> int:
        with self.session() as s:
            for row in rows:
                s.merge(Sample(sample_id=row["sample_id"], project_id=row["project_id"],
                               group=row.get("group", ""), batch=row.get("batch", ""),
                               sample_type=row.get("sample_type", ""),
                               tenant_id=row.get("tenant_id", "mihc")))
            s.commit()
        return len(rows)

    def list_samples(self, project_id: str, tenant_id: str = "mihc") -> List[dict]:
        with self.session() as s:
            rows = s.query(Sample).filter(Sample.project_id == project_id, Sample.tenant_id == tenant_id).all()
            return [{"sample_id": r.sample_id, "project_id": r.project_id, "group": r.group,
                     "batch": r.batch, "sample_type": r.sample_type} for r in rows]

    def add_marker(self, *, marker_id: str, project_id: str, name: str, channel: str = "",
                   antibody: str = "", threshold: Optional[float] = None, unit: str = "",
                   tenant_id: str = "mihc") -> dict:
        with self.session() as s:
            row = Marker(marker_id=marker_id, project_id=project_id, name=name, channel=channel,
                         antibody=antibody, threshold=threshold, unit=unit, tenant_id=tenant_id)
            s.add(row)
            s.commit()
            return {"marker_id": row.marker_id, "project_id": row.project_id, "name": row.name,
                    "channel": row.channel, "antibody": row.antibody, "threshold": row.threshold,
                    "unit": row.unit}

    def list_markers(self, project_id: str, tenant_id: str = "mihc") -> List[dict]:
        with self.session() as s:
            rows = s.query(Marker).filter(Marker.project_id == project_id, Marker.tenant_id == tenant_id).all()
            return [{"marker_id": r.marker_id, "project_id": r.project_id, "name": r.name,
                     "channel": r.channel, "antibody": r.antibody, "threshold": r.threshold,
                     "unit": r.unit} for r in rows]

    def add_project_file(self, **kwargs) -> dict:
        with self.session() as s:
            row = ProjectFile(**kwargs)
            s.add(row)
            s.commit()
            return self._file_dict(row)

    @staticmethod
    def _file_dict(row: ProjectFile) -> dict:
        return {"file_id": row.file_id, "project_id": row.project_id, "tenant_id": row.tenant_id,
                "file_name": row.file_name, "object_key": row.object_key,
                "content_type": row.content_type, "extension": row.extension, "kind": row.kind,
                "size_bytes": row.size_bytes, "sha256": row.sha256, "status": row.status,
                "created_by": row.created_by,
                "created_at": row.created_at.isoformat() if row.created_at else ""}

    def get_project_file(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[dict]:
        with self.session() as s:
            row = s.get(ProjectFile, file_id)
            if not row or (tenant_id and row.tenant_id != tenant_id):
                return None
            return self._file_dict(row)

    def list_project_files(self, project_id: str, tenant_id: str = "mihc") -> List[dict]:
        with self.session() as s:
            rows = s.query(ProjectFile).filter(ProjectFile.project_id == project_id,
                                                ProjectFile.tenant_id == tenant_id).order_by(ProjectFile.created_at.desc()).all()
            return [self._file_dict(row) for row in rows]

    @staticmethod
    def _run_dict(row: AnalysisRun) -> dict:
        return {"run_id": row.run_id, "project_id": row.project_id, "tenant_id": row.tenant_id,
                "user_id": row.user_id, "branch": row.branch, "status": row.status,
                "current_node": row.current_node, "input_refs": Database._json_load(row.input_refs, []),
                "output_refs": Database._json_load(row.output_refs, []),
                "config": Database._json_load(row.config_json, {}),
                "result": Database._json_load(row.result_json, {}),
                "errors": Database._json_load(row.errors_json, []), "next_action": row.next_action,
                "model_version": row.model_version,
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "updated_at": row.updated_at.isoformat() if row.updated_at else ""}

    def create_analysis_run(self, *, run_id: str, project_id: str, tenant_id: str, user_id: str,
                            branch: str, status: str = "received", current_node: str = "received",
                            input_refs: Optional[List[str]] = None, config: Optional[dict] = None,
                            model_version: str = "") -> dict:
        with self.session() as s:
            row = AnalysisRun(run_id=run_id, project_id=project_id, tenant_id=tenant_id, user_id=user_id,
                              branch=branch, status=status, current_node=current_node,
                              input_refs=json.dumps(input_refs or [], ensure_ascii=False),
                              config_json=json.dumps(config or {}, ensure_ascii=False),
                              model_version=model_version)
            s.add(row)
            s.commit()
            return self._run_dict(row)

    def update_analysis_run(self, run_id: str, **kwargs) -> Optional[dict]:
        with self.session() as s:
            row = s.get(AnalysisRun, run_id)
            if not row:
                return None
            mapping = {"input_refs": "input_refs", "output_refs": "output_refs", "config": "config_json",
                       "result": "result_json", "errors": "errors_json"}
            for key, value in kwargs.items():
                attr = mapping.get(key, key)
                if key in mapping:
                    value = json.dumps(value or ([] if key.endswith("s") else {}), ensure_ascii=False)
                if hasattr(row, attr):
                    setattr(row, attr, value)
            s.commit()
            return self._run_dict(row)

    def get_analysis_run(self, run_id: str, tenant_id: Optional[str] = None) -> Optional[dict]:
        with self.session() as s:
            row = s.get(AnalysisRun, run_id)
            if not row or (tenant_id and row.tenant_id != tenant_id):
                return None
            return self._run_dict(row)

    def add_report(self, **kwargs) -> dict:
        with self.session() as s:
            row = Report(**kwargs)
            s.add(row)
            s.commit()
            return {"report_id": row.report_id, "project_id": row.project_id, "run_id": row.run_id,
                    "status": row.status, "template_version": row.template_version,
                    "json_object_key": row.json_object_key, "html_object_key": row.html_object_key,
                    "review_status": row.review_status}

    def get_report(self, report_id: str, tenant_id: Optional[str] = None) -> Optional[dict]:
        with self.session() as s:
            row = s.get(Report, report_id)
            if not row or (tenant_id and row.tenant_id != tenant_id):
                return None
            return {"report_id": row.report_id, "project_id": row.project_id, "run_id": row.run_id,
                    "tenant_id": row.tenant_id, "status": row.status, "template_version": row.template_version,
                    "json_object_key": row.json_object_key, "html_object_key": row.html_object_key,
                    "review_status": row.review_status,
                    "created_at": row.created_at.isoformat() if row.created_at else ""}

    def get_user(self, username: str) -> Optional[dict]:
        with self.session() as s:
            row = s.query(User).filter(User.username == username).first()
            if not row:
                return None
            return {"id": row.id, "username": row.username, "password_hash": row.password_hash,
                    "tenant_id": row.tenant_id, "role": row.role,
                    "display_name": row.display_name or row.username}

    def create_user(self, *, username: str, password_hash: str, tenant_id: str = "mihc",
                    role: str = "customer", display_name: str = "") -> dict:
        with self.session() as s:
            row = User(username=username, password_hash=password_hash, tenant_id=tenant_id,
                       role=role, display_name=display_name or username)
            s.add(row)
            s.commit()
            return {"id": row.id, "username": row.username, "tenant_id": row.tenant_id,
                    "role": row.role, "display_name": row.display_name or row.username}

    def add_chunks(self, chunks: List[dict]) -> None:
        with self.session() as s:
            for c in chunks:
                s.add(Chunk(chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c.get("title", ""),
                            text_snippet=c.get("text", "")[:2000], source=c.get("source", ""),
                            version=c.get("version", ""), tenant_id=c.get("tenant_id", "mihc"),
                            embedding_model=c.get("embedding_model", "BGE-M3")))
            s.commit()

    def clear_knowledge(self) -> None:
        with self.session() as s:
            s.query(Chunk).delete()
            s.query(Document).delete()
            s.commit()

    def add_audit(self, **kwargs) -> None:
        with self.session() as s:
            s.add(AuditLog(**kwargs))
            s.commit()

    def add_badcase(self, **kwargs) -> int:
        with self.session() as s:
            obj = Badcase(**kwargs)
            s.add(obj)
            s.commit()
            return obj.id

    def list_badcases(self, stage: Optional[str] = None, limit: int = 100) -> List[dict]:
        with self.session() as s:
            q = s.query(Badcase).order_by(Badcase.created_at.desc()).limit(limit)
            if stage:
                q = q.filter(Badcase.stage == stage)
            return [
                {"id": r.id, "trace_id": r.trace_id, "query": r.query, "answer": r.answer,
                 "intent": r.intent, "stage": r.stage, "feedback": r.feedback,
                 "expert_label": r.expert_label, "root_cause": r.root_cause,
                 "created_at": r.created_at.isoformat() if r.created_at else ""}
                for r in q.all()
            ]
