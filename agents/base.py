"""
Agent 基类与共享 RAG 生成能力

设计要点（对齐《项目文档》5.5）：
- 每个 Agent 独立上下文（agent_outputs 隔离），只把必要结果合并；
- 工具超时、参数校验、失败重试（最多 2 次）；
- 生成遵守证据约束：只用检索片段、给出引用、不确定时说明证据不足。
"""

from __future__ import annotations

import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from core.errors import AgentError
from core.models import Citation, Risk

logger = logging.getLogger(__name__)

EVIDENCE_PROMPT = """你是 MIHC 医疗科研智能平台的{role_desc}。

<rules>
必须严格遵守以下规则：
1. 只使用 <evidence> 中提供的证据片段回答，不得使用内部知识编造；
2. 每个关键结论必须标注引用编号，如[1][2]；
3. 检索证据不足时，明确说明"证据不足"，不要猜测；
4. 医疗建议与科研建议分开表达；不要把实验假设写成临床结论；
5. 回答使用中文，结构清晰。
</rules>

<query>
{query}
</query>

<evidence>
{context}
</evidence>

请作答："""


class BaseAgent:
    """专业 Agent 基类：轨迹记录 + 重试封装。"""

    name: str = "base_agent"
    max_retries: int = 2

    def __init__(self, config, llm_factory):
        self.config = config
        self.llm_factory = llm_factory

    def run(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        """执行一个规划步骤，返回 {output, citations, risks}。子类实现 _execute。"""
        start = time.time()
        agent_name = self.name
        try:
            result = self._execute(state, step)
            latency = int((time.time() - start) * 1000)
            state.setdefault("agent_traces", []).append({
                "agent": agent_name, "status": "ok", "latency_ms": latency,
                "detail": str(step.get("description", ""))[:120],
            })
            logger.info("[%s] done in %dms", agent_name, latency)
            return result
        except Exception as exc:  # noqa: BLE001
            # 失败重试（最多 max_retries 次）
            for attempt in range(1, self.max_retries + 1):
                logger.warning("[%s] attempt %d failed: %s, retry %d", agent_name, attempt, exc, attempt + 1)
                try:
                    result = self._execute(state, step)
                    latency = int((time.time() - start) * 1000)
                    state.setdefault("agent_traces", []).append({
                        "agent": agent_name, "status": "ok", "latency_ms": latency,
                        "detail": f"retried after {attempt} failure",
                    })
                    return result
                except Exception as retry_exc:  # noqa: BLE001
                    exc = retry_exc
            latency = int((time.time() - start) * 1000)
            state.setdefault("agent_traces", []).append({
                "agent": agent_name, "status": "failed", "latency_ms": latency,
                "detail": str(exc)[:120],
            })
            return {
                "output": f"（{agent_name} 执行失败：{exc}）",
                "citations": [],
                "risks": [{"level": "warning", "message": f"{agent_name} 执行失败，已记录审计日志"}],
            }

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def build_evidence_context(chunks: List[Dict[str, Any]], max_chars: int = 8192) -> Tuple[str, List[Citation]]:
    """把检索片段拼成带编号的上下文（截断保护），同时生成引用列表。"""
    citations: List[Citation] = []
    parts: List[str] = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        snippet = c.get("text", "")
        if total + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - total)]
        if not snippet:
            continue
        parts.append(f"[{i}] {snippet}")
        total += len(snippet)
        citations.append(Citation(
            chunk_id=c.get("chunk_id", ""),
            title=c.get("title", ""),
            source=c.get("source", ""),
            text_snippet=snippet[:500],
            score=round(float(c.get("rerank_score", c.get("score", 0.0))), 4),
        ))
        if total >= max_chars:
            break
    return "\n\n".join(parts), citations


def generate_with_evidence(llm_factory, query: str, chunks: List[Dict[str, Any]],
                           role_desc: str, max_context_chars: int = 8192) -> Dict[str, Any]:
    """证据约束生成：返回 {answer, citations, confidence}。"""
    context, citations = build_evidence_context(chunks, max_context_chars)
    if not citations:
        return {
            "answer": "当前知识库中未检索到与该问题相关的证据，无法给出有依据的回答。",
            "citations": [],
            "confidence": 0.0,
        }
    prompt = EVIDENCE_PROMPT.format(role_desc=role_desc, query=query, context=context)
    answer = llm_factory.invoke_with_failover(
        [{"role": "user", "content": prompt}],
        role="rag_generate",
    )
    # 检索置信度：重排分均值归一化（与旧系统一致的近似口径）
    scores = [float(c.get("rerank_score", 0.0)) for c in chunks if c.get("rerank_score") is not None]
    confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {"answer": answer, "citations": citations, "confidence": confidence}


def strip_risk(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
