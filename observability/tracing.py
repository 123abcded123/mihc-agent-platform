"""
链路追踪封装（OTel span + Langfuse 可选）

- 每个请求/每个 Agent 执行一个 span，记录意图、耗时、状态；
- Langfuse 三个密钥齐全时启用，记录 LLM 级别输入输出与 token（对齐文档 5.8）。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_langfuse = None


def init_langfuse(config) -> None:
    global _langfuse
    if not (config.observability.langfuse_public_key and config.observability.langfuse_secret_key):
        return
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=config.observability.langfuse_public_key,
            secret_key=config.observability.langfuse_secret_key,
            host=config.observability.langfuse_host,
        )
        logger.info("Langfuse tracing enabled")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse init failed: %s", exc)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """OTel span 上下文管理器（记录耗时与异常）。"""
    from observability.otel import get_tracer
    tracer = get_tracer()
    start = time.time()
    with tracer.start_as_current_span(name) as s:
        for k, v in (attributes or {}).items():
            s.set_attribute(k, v)
        try:
            yield s
        except Exception as exc:  # noqa: BLE001
            s.record_exception(exc)
            raise
        finally:
            s.set_attribute("latency_ms", int((time.time() - start) * 1000))


def log_generation(trace_id: str, name: str, prompt: str, completion: str,
                   model: str = "", latency_ms: int = 0) -> None:
    """可选 Langfuse 生成记录（LLM 级别）。"""
    if _langfuse is None:
        return
    try:
        _langfuse.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            input={"prompt": prompt[:2000]},
            output={"completion": completion[:2000]},
            metadata={"latency_ms": latency_ms},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse log failed: %s", exc)
