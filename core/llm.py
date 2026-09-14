"""
LLM 模型工厂与故障切换模块

职责：
1. 统一封装"vLLM 服务（Qwen3-32B，LoRA 微调）+ OpenAI 兼容端点"两种后端，
   对上层暴露同一种调用方式；
2. 按任务角色返回不同 temperature 的模型实例（路由决策 0.1、闲聊 0.7 等）；
3. 模型网关能力：候选模型列表 + 超时切换备用（对齐模型网关教学伪代码）。

配置来源：core.config 中的 llm 配置块。
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from core.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

# 角色 -> temperature 映射（温度越低越确定）
ROLE_TEMPERATURE = {
    "decision": 0.1,          # 意图路由/决策：确定性输出
    "planner": 0.2,           # 任务规划：稳定拆解
    "conversation": 0.7,      # 闲聊对话：自然但有事实约束
    "search_summary": 0.3,    # 文献/检索结果总结
    "rag_generate": 0.3,      # RAG 答案生成
    "summarizer": 0.5,        # 图片/内容摘要
    "chunker": 0.0,           # 文档语义切块：完全确定
    "hyde": 0.4,              # 假设文档生成
    "rewrite": 0.2,           # 查询改写
    "citation_check": 0.0,    # 引用检查
    "guard": 0.0,             # 安全护栏
}


def _build_chat_openai(base_url: Optional[str], api_key: Optional[str], model: str, temperature: float) -> ChatOpenAI:
    """按端点构建 LangChain ChatOpenAI（OpenAI 兼容协议）。"""
    kwargs: Dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        api_key=api_key or "EMPTY",  # vLLM 部署通常不校验 key
        request_timeout=180,
        max_retries=1,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


class LLMFactory:
    """
    LLM 实例工厂。

    选择逻辑：
    - LLM_BASE_URL 已配置 → 主用 vLLM 端点（模型名如 qwen3-32b-lora）；
    - 未配置 → 主用 OpenAI 官方端点（模型名如 gpt-4o）；
    - 任一后端超时/失败 → 按 candidates 顺序切换备用端点（模型网关故障切换）。
    """

    def __init__(self, config):
        self.config = config
        self.primary = {
            "base_url": config.llm_base_url or None,
            "api_key": config.llm_api_key,
            "model": config.llm_model,
        }
        # 备用端点：显式配置 LLM_FALLBACK_BASE_URL 时使用；
        # 未配置且主端点就是 OpenAI 官方时，无第二候选（避免把代理密钥发到官方端点）
        self.fallback = {
            "base_url": config.llm_fallback_base_url or None,
            "api_key": config.llm_fallback_api_key or config.llm_api_key,
            "model": config.llm_fallback_model or config.llm_model,
        }
        self._instances: Dict[str, BaseChatModel] = {}

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        """候选端点列表（去重；备用端点未显式配置 base_url 时不参与，避免密钥泄漏到错误端点）。"""
        seen, out = set(), []
        for ep in (self.primary, self.fallback):
            key = (ep["base_url"], ep["model"])
            if key in seen:
                continue
            if ep is self.fallback and not ep["base_url"]:
                continue  # 未配置备用端点 → 跳过
            seen.add(key)
            out.append(ep)
        return out

    def get_llm(self, role: str = "rag_generate", temperature: Optional[float] = None) -> BaseChatModel:
        """
        获取指定角色的 LLM 实例（带缓存）。
        temperature 未显式指定时按角色默认值。
        """
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
        """
        模型网关调用：依次尝试候选端点，超时/失败自动切换，全部失败抛 ModelUnavailableError。
        """
        temp = temperature if temperature is not None else ROLE_TEMPERATURE.get(role, 0.3)
        last_error: Optional[Exception] = None
        for ep in self.candidates:
            try:
                client = _build_chat_openai(
                    base_url=ep["base_url"], api_key=ep["api_key"], model=ep["model"], temperature=temp
                )
                resp = client.invoke(messages)
                return str(resp.content)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM endpoint %s failed: %s", ep["model"], exc)
                last_error = exc
        raise ModelUnavailableError(f"所有模型服务均不可用: {last_error}")
