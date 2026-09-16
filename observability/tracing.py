
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
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)

def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]

@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    from observability.otel import get_tracer
    tracer = get_tracer()
    start = time.time()
    with tracer.start_as_current_span(name) as s:
        for k, v in (attributes or {}).items():
            s.set_attribute(k, v)
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            raise
        finally:
            s.set_attribute("latency_ms", int((time.time() - start) * 1000))

def log_generation(trace_id: str, name: str, prompt: str, completion: str,
                   model: str = "", latency_ms: int = 0) -> None:
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
    except Exception as exc:
        logger.debug("Langfuse log failed: %s", exc)
