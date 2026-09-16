
from __future__ import annotations

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """你是医疗科研平台的智能任务规划器。请把用户问题拆解为可执行的步骤。

可用 Agent：
- literature_agent 文献检索：检索文献、整理证据、输出引用
- knowledge_agent 知识问答：基于内部知识库回答医学知识
- data_analysis_agent 数据分析：统计方法、数据处理、图表分析
- experiment_agent 实验设计：设计实验方案并校验

要求：
1. 简单问题可以只有 1 个步骤；复杂问题最多 5 个步骤；
2. 每一步要可验证（check 字段描述验证方式，如"确认引用年份≥2020"）；
3. 实验设计步骤必须包含方案校验：对照组、重复数、终点指标；
4. 只输出 JSON 数组，不要任何其他文字。

格式：
[{{"step_id": 1, "description": "检索相关文献", "agent": "literature_agent", "check": "引用数量≥2且年份符合要求"}}]

用户问题：{query}"""

class TaskPlanner:

    def __init__(self, config, llm_factory):
        self.config = config
        self.llm_factory = llm_factory
        self.max_steps = config.platform.max_agent_steps

    def plan(self, query: str) -> List[Dict[str, Any]]:
        try:
            raw = self.llm_factory.invoke_with_failover(
                [{"role": "user", "content": PLANNER_PROMPT.format(query=query)}],
                role="planner",
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            steps = json.loads(raw)
            if not isinstance(steps, list) or not steps:
                raise ValueError("empty plan")
        except Exception as exc:
            logger.warning("Planner failed (%s), fallback to single-step plan", exc)
            steps = [{"step_id": 1, "description": query, "agent": self._default_agent(query), "check": ""}]

        valid = []
        for i, s in enumerate(steps[: self.max_steps], start=1):
            valid.append({
                "step_id": int(s.get("step_id", i)),
                "description": str(s.get("description", "")).strip() or query,
                "agent": str(s.get("agent", "")).strip() or "knowledge_agent",
                "check": str(s.get("check", "")).strip(),
            })
        logger.info("Plan: %s", [(s["agent"], s["description"][:40]) for s in valid])
        return valid

    @staticmethod
    def _default_agent(query: str) -> str:
        q = query
        for kw in ("检索", "文献", "论文", "研究进展"):
            if kw in q:
                return "literature_agent"
        for kw in ("实验", "方案", "验证", "设计"):
            if kw in q:
                return "experiment_agent"
        for kw in ("统计", "数据分析", "差异", "相关性"):
            if kw in q:
                return "data_analysis_agent"
        return "knowledge_agent"
