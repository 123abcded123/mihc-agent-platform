
from __future__ import annotations

import logging
from typing import Dict, Any

from agents.base import BaseAgent, generate_with_evidence

logger = logging.getLogger(__name__)

class KnowledgeAgent(BaseAgent):
    name = "knowledge_agent"

    def __init__(self, config, llm_factory, retriever, query_processor):
        super().__init__(config, llm_factory)
        self.retriever = retriever
        self.query_processor = query_processor

    def _execute(self, state: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        query = step.get("description") or state.get("query", "")
        queries = self.query_processor.build_queries(query)
        chunks = self.retriever.retrieve(
            queries,
            tenant_id=state.get("tenant_id", "mihc"),
        )
        result = generate_with_evidence(
            self.llm_factory,
            query,
            chunks,
            role_desc="医疗科研知识问答专家",
            max_context_chars=self.config.retrieval.max_context_chars,
        )
        result["risks"] = [{
            "level": "info",
            "message": "本回答仅用于科研参考，不构成临床诊疗建议",
        }]
        if result["confidence"] < self.config.retrieval.min_retrieval_confidence:
            result["risks"].append({
                "level": "warning",
                "message": f"检索置信度较低({result['confidence']:.2f})，结论可靠性有限",
            })
        return {
            "output": result["answer"],
            "citations": [c.model_dump() for c in result["citations"]],
            "risks": result["risks"],
            "retrieval_confidence": result["confidence"],
        }
