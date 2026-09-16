
from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from core.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

ROLE_TEMPERATURE = {
    "decision": 0.1,
    "planner": 0.2,
    "conversation": 0.7,
    "search_summary": 0.3,
    "rag_generate": 0.3,
    "summarizer": 0.5,
    "chunker": 0.0,
    "hyde": 0.4,
    "rewrite": 0.2,
    "citation_check": 0.0,
    "guard": 0.0,
}

def _build_chat_openai(base_url: Optional[str], api_key: Optional[str], model: str, temperature: float) -> ChatOpenAI:
    kwargs: Dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        api_key=api_key or "EMPTY",
        request_timeout=180,
        max_retries=1,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)

class LLMFactory:

    def __init__(self, config):
        self.config = config
        self.primary = {
            "base_url": config.llm_base_url or None,
            "api_key": config.llm_api_key,
            "model": config.llm_model,
        }
        self.fallback = {
            "base_url": config.llm_fallback_base_url or None,
            "api_key": config.llm_fallback_api_key or config.llm_api_key,
            "model": config.llm_fallback_model or config.llm_model,
        }
        self._instances: Dict[str, BaseChatModel] = {}

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for ep in (self.primary, self.fallback):
            key = (ep["base_url"], ep["model"])
            if key in seen:
                continue
            if ep is self.fallback and not ep["base_url"]:
                continue
            seen.add(key)
            out.append(ep)
        return out

    def get_llm(self, role: str = "rag_generate", temperature: Optional[float] = None) -> BaseChatModel:
        temp = temperature if temperature is not None else ROLE_TEMPERATURE.get(role, 0.3)
        cache_key = f"{role}:{temp}"
        if cache_key not in self._instances:
            ep = self.candidates[0]
            self._instances[cache_key] = _build_chat_openai(
                base_url=ep["base_url"], api_key=ep["api_key"], model=ep["model"], temperature=temp
            )
            logger.info("LLM instance [%s] -> %s @ %s", cache_key, ep["model"], ep["base_url"] or "openai")
        return self._instances[cache_key]

    def invoke_with_failover(self, messages: List[Dict[str, str]], role: str = "rag_generate",
                             temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else ROLE_TEMPERATURE.get(role, 0.3)
        last_error: Optional[Exception] = None
        for ep in self.candidates:
            try:
                client = _build_chat_openai(
                    base_url=ep["base_url"], api_key=ep["api_key"], model=ep["model"], temperature=temp
                )
                resp = client.invoke(messages)
                return str(resp.content)
            except Exception as exc:
                logger.warning("LLM endpoint %s failed: %s", ep["model"], exc)
                last_error = exc
        raise ModelUnavailableError(f"所有模型服务均不可用: {last_error}")
