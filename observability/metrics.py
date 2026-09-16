
from __future__ import annotations

import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

REQUEST_TOTAL = Counter(
    "mihc_request_total", "Total chat requests", ["intent", "status"]
)
REQUEST_LATENCY = Histogram(
    "mihc_request_latency_seconds", "Request latency", ["intent"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 120, 240),
)
AGENT_DURATION = Histogram(
    "mihc_agent_duration_seconds", "Agent execution duration", ["agent"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40, 80),
)
RETRIEVAL_CANDIDATES = Histogram(
    "mihc_retrieval_candidates", "Candidates after RRF fusion",
    buckets=(1, 3, 5, 10, 20, 50),
)
GUARD_BLOCKS = Counter(
    "mihc_guard_blocks_total", "Guardrail blocks", ["side"]
)
LLM_TOKENS = Counter(
    "mihc_llm_tokens_total", "Estimated LLM tokens", ["kind"]
)
ACTIVE_SESSIONS = Gauge("mihc_active_sessions", "Active sessions (estimate)")

def record_request(intent: str, status: str, latency_s: float) -> None:
    REQUEST_TOTAL.labels(intent=intent, status=status).inc()
    REQUEST_LATENCY.labels(intent=intent).observe(latency_s)

def record_agent(agent: str, duration_s: float) -> None:
    AGENT_DURATION.labels(agent=agent).observe(duration_s)

def record_retrieval(candidates: int) -> None:
    RETRIEVAL_CANDIDATES.observe(max(0, candidates))

def record_guard_block(side: str) -> None:
    GUARD_BLOCKS.labels(side=side).inc()

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)
