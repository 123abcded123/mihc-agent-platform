
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, START, END

from core.errors import InputGuardError, OutputGuardError
from core.llm import LLMFactory
from core.embeddings import EmbedderFactory
from agents.state import AgentState
from agents.planner import TaskPlanner
from agents.intent.bert_intent import IntentClassifier
from agents.base import BaseAgent
from agents.literature_agent import LiteratureAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.data_analysis_agent import DataAnalysisAgent
from agents.experiment_agent import ExperimentAgent
from agents.merge import merge_outputs
from agents.citation_check import check_citations
from rag.query_processor import QueryProcessor
from rag.retriever.hybrid_retriever import HybridRetriever
from security.input_guard import InputGuard
from security.prompt_guard import PromptGuardrail
from security.dfa import DFASensitiveFilter
from observability import metrics as obs_metrics

logger = logging.getLogger(__name__)

CONVERSATION_PROMPT = """你是 MIHC 医疗科研智能平台的科研助手。用户问题是闲聊或非科研问题。
要求：礼貌、简洁、中文回答；如果是医疗咨询，提醒平台定位为科研服务，不提供个人诊疗。"""

REFUSAL_MESSAGE = "抱歉，您的输入包含敏感内容或不符合平台使用规范，本次请求已被安全系统拦截。"

class AgentGraph:

    def __init__(self, config):
        self.config = config
        self._llm_factory: Optional[LLMFactory] = None
        self._embedder: Optional[Any] = None
        self._retriever: Optional[HybridRetriever] = None
        self._query_processor: Optional[QueryProcessor] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._planner: Optional[TaskPlanner] = None
        self._input_guard: Optional[InputGuard] = None
        self._output_guard: Optional[PromptGuardrail] = None
        self._output_dfa: Optional[DFASensitiveFilter] = None
        self._agents: Dict[str, BaseAgent] = {}

    @property
    def llm(self) -> LLMFactory:
        if self._llm_factory is None:
            self._llm_factory = LLMFactory(self.config)
        return self._llm_factory

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbedderFactory(self.config).get()
        return self._embedder

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.config, embedder=self.embedder)
        return self._retriever

    @property
    def query_processor(self) -> QueryProcessor:
        if self._query_processor is None:
            self._query_processor = QueryProcessor(self.config, self.llm)
        return self._query_processor

    @property
    def intent_classifier(self) -> IntentClassifier:
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier(self.config, self.llm)
        return self._intent_classifier

    @property
    def planner(self) -> TaskPlanner:
        if self._planner is None:
            self._planner = TaskPlanner(self.config, self.llm)
        return self._planner

    @property
    def input_guard(self) -> InputGuard:
        if self._input_guard is None:
            self._input_guard = InputGuard(self.config, self.llm)
            self._input_guard.prompt_guard.enabled = (
                self.config.security.enable_prompt_guard and not self.config.platform.fast_mode)
        return self._input_guard

    @property
    def output_guard(self) -> PromptGuardrail:
        if self._output_guard is None:
            self._output_guard = PromptGuardrail(
                self.llm,
                enabled=self.config.security.enable_output_scan and not self.config.platform.fast_mode)
        return self._output_guard

    @property
    def output_dfa(self) -> DFASensitiveFilter:
        if self._output_dfa is None:
            self._output_dfa = DFASensitiveFilter(self.config.security.dfa_words_file)
        return self._output_dfa

    def get_agents(self) -> Dict[str, BaseAgent]:
        if not self._agents:
            self._agents = {
                "literature_agent": LiteratureAgent(
                    self.config, self.llm, self.retriever, self.query_processor),
                "knowledge_agent": KnowledgeAgent(
                    self.config, self.llm, self.retriever, self.query_processor),
                "data_analysis_agent": DataAnalysisAgent(
                    self.config, self.llm, self.retriever, self.query_processor),
                "experiment_agent": ExperimentAgent(
                    self.config, self.llm, self.retriever, self.query_processor),
            }
        return self._agents

    def _node_load_context(self, state: AgentState) -> Dict[str, Any]:
        return {}

    def _node_input_guard(self, state: AgentState) -> Dict[str, Any]:
        allowed, reason = self.input_guard.check(state.get("query", ""))
        if not allowed:
            obs_metrics.record_guard_block("input")
            return {
                "guard_blocked": True,
                "guard_reason": reason,
                "answer": REFUSAL_MESSAGE,
                "merged_output": REFUSAL_MESSAGE,
                "risks": [{"level": "warning", "message": reason}],
            }
        return {"guard_blocked": False, "guard_reason": ""}

    def _node_intent(self, state: AgentState) -> Dict[str, Any]:
        result = self.intent_classifier.classify(state.get("query", ""))
        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
            "intent_model": result["model"],
        }

    def _node_conversation(self, state: AgentState) -> Dict[str, Any]:
        history = state.get("history", [])[-6:]
        system_prompt = CONVERSATION_PROMPT
        project_context = state.get("project_context", "")
        if project_context:
            system_prompt += f"\n{project_context}，回答与项目相关的内容时可结合项目编号。"
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": state.get("query", "")})
        answer = self.llm.invoke_with_failover(messages, role="conversation")
        return {"answer": answer, "merged_output": answer}

    def _node_plan(self, state: AgentState) -> Dict[str, Any]:
        if self.config.platform.fast_mode:
            query = state.get("query", "")
            step = {"step_id": 1, "description": query,
                    "agent": TaskPlanner._default_agent(query), "check": ""}
            return {"plan": [step], "step_index": 0, "max_steps": 1}
        query = state.get("query", "")
        if state.get("project_context"):
            query = f"{query}\n（{state['project_context']}）"
        plan = self.planner.plan(query)
        return {"plan": plan, "step_index": 0, "max_steps": len(plan)}

    def _node_execute_step(self, state: AgentState) -> Dict[str, Any]:
        step_index = state.get("step_index", 0)
        plan = state.get("plan", [])
        if step_index >= len(plan):
            return {}
        step = plan[step_index]
        agent_name = step.get("agent", "knowledge_agent")
        agents = self.get_agents()
        agent = agents.get(agent_name) or agents["knowledge_agent"]

        result = agent.run(state, step)
        outputs = dict(state.get("agent_outputs", {}))
        outputs[str(step.get("step_id"))] = result

        agent_chain = list(state.get("agent_chain", []))
        if agent_name not in agent_chain:
            agent_chain.append(agent_name)

        return {
            "agent_outputs": outputs,
            "agent_chain": agent_chain,
            "step_index": step_index + 1,
        }

    def _node_merge(self, state: AgentState) -> Dict[str, Any]:
        return merge_outputs(state)

    def _node_citation_check(self, state: AgentState) -> Dict[str, Any]:
        return check_citations(state, self.llm, fast_mode=self.config.platform.fast_mode)

    def _node_output_guard(self, state: AgentState) -> Dict[str, Any]:
        answer = state.get("answer", state.get("merged_output", ""))
        hits = self.output_dfa.scan(answer)
        if hits:
            obs_metrics.record_guard_block("output")
            return {
                "answer": f"{answer}\n\n> ⚠ 输出安全检查：检测到敏感内容，请人工复核。",
                "output_blocked": True,
            }
        safe, reason = self.output_guard.check_output(answer)
        if not safe:
            obs_metrics.record_guard_block("output")
            logger.info("Output flagged by guard: %s", reason)
            return {
                "answer": f"{answer}\n\n> ⚠ 输出安全检查提示：{reason or '内容需人工复核'}。",
                "output_blocked": True,
            }
        return {"answer": answer, "output_blocked": False}

    @staticmethod
    def _after_guard(state: AgentState) -> str:
        return "end" if state.get("guard_blocked") else "classify_intent"

    @staticmethod
    def _after_intent(state: AgentState) -> str:
        return "conversation" if state.get("intent") == "other" else "build_plan"

    @staticmethod
    def _after_step(state: AgentState) -> str:
        return "execute_step" if state.get("step_index", 0) < len(state.get("plan", [])) else "merge"

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_context", self._node_load_context)
        graph.add_node("input_guard", self._node_input_guard)
        graph.add_node("classify_intent", self._node_intent)
        graph.add_node("conversation", self._node_conversation)
        graph.add_node("build_plan", self._node_plan)
        graph.add_node("execute_step", self._node_execute_step)
        graph.add_node("merge", self._node_merge)
        graph.add_node("citation_check", self._node_citation_check)
        graph.add_node("output_guard", self._node_output_guard)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "input_guard")
        graph.add_conditional_edges("input_guard", self._after_guard, {
            "classify_intent": "classify_intent",
            "end": END,
        })
        graph.add_conditional_edges("classify_intent", self._after_intent, {
            "conversation": "conversation",
            "build_plan": "build_plan",
        })
        graph.add_edge("conversation", "output_guard")
        graph.add_edge("build_plan", "execute_step")
        graph.add_conditional_edges("execute_step", self._after_step, {
            "execute_step": "execute_step",
            "merge": "merge",
        })
        graph.add_edge("merge", "citation_check")
        graph.add_edge("citation_check", "output_guard")
        graph.add_edge("output_guard", END)
        return graph.compile()

    def warmup(self):
        try:
            logger.info("Warmup: loading embedding model ...")
            _ = self.embedder
            logger.info("Warmup: loading reranker model ...")
            _ = self.retriever
            logger.info("Warmup: done")
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)
