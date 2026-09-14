"""
输入侧安全门（DFA 敏感词 + Prompt Guardrail 组合）

对齐《项目文档》5.7：DFA 不能识别所有隐含风险，需要输入侧敏感信息检测
与 Prompt 注入防护组合使用。
"""

from __future__ import annotations

import logging
from typing import Tuple

from security.dfa import DFASensitiveFilter
from security.prompt_guard import PromptGuardrail

logger = logging.getLogger(__name__)


class InputGuard:
    """输入侧护栏门面。"""

    def __init__(self, config, llm_factory):
        self.config = config
        self.dfa = DFASensitiveFilter(config.security.dfa_words_file)
        self.prompt_guard = PromptGuardrail(llm_factory, enabled=config.security.enable_prompt_guard)
        self.enabled = config.security.enable_input_scan

    def check(self, text: str) -> Tuple[bool, str]:
        """返回 (allowed, reason)。"""
        if not self.enabled:
            return True, ""

        # 第一层：DFA 明文敏感词
        hits = self.dfa.scan(text)
        if hits:
            reason = f"输入包含敏感内容: {', '.join(hits[:3])}"
            logger.info("Input blocked by DFA: %s", reason)
            return False, reason

        # 第二层：Prompt 注入/隐含风险（LLM）
        safe, reason = self.prompt_guard.check_input(text)
        if not safe:
            logger.info("Input blocked by prompt guard: %s", reason)
            return False, reason or "输入未通过安全审查"

        return True, ""
