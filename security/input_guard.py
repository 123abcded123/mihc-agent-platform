
from __future__ import annotations

import logging
from typing import Tuple

from security.dfa import DFASensitiveFilter
from security.prompt_guard import PromptGuardrail

logger = logging.getLogger(__name__)

class InputGuard:

    def __init__(self, config, llm_factory):
        self.config = config
        self.dfa = DFASensitiveFilter(config.security.dfa_words_file)
        self.prompt_guard = PromptGuardrail(llm_factory, enabled=config.security.enable_prompt_guard)
        self.enabled = config.security.enable_input_scan

    def check(self, text: str) -> Tuple[bool, str]:
        if not self.enabled:
            return True, ""

        hits = self.dfa.scan(text)
        if hits:
            reason = f"输入包含敏感内容: {', '.join(hits[:3])}"
            logger.info("Input blocked by DFA: %s", reason)
            return False, reason

        safe, reason = self.prompt_guard.check_input(text)
        if not safe:
            logger.info("Input blocked by prompt guard: %s", reason)
            return False, reason or "输入未通过安全审查"

        return True, ""
