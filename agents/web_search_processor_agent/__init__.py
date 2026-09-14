"""
web_search_processor_agent 包：联网搜索处理 Agent 模块。

对外暴露 WebSearchProcessorAgent，作为“联网搜索”能力的统一入口：
接收用户问题与最近对话历史，内部完成
“LLM 压缩查询 → 搜索引擎检索（Tavily）→ LLM 汇总生成医学总结”的完整链路。
"""
from typing import List, Dict, Any, Optional
from .web_search_processor import WebSearchProcessor

class WebSearchProcessorAgent:
    """
    联网搜索处理 Agent（门面类）。

    负责把联网搜索结果加工成适合展示给用户的医学回复，
    供上层多智能体路由器（如 RAG 置信度不足时的兜底）调用。

    原英文注释：
    Agent responsible for processing web search results and routing them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        """初始化底层 WebSearchProcessor 处理器。
        参数 config：全局配置对象（其中含 web_search.llm 等联网搜索相关配置）。"""
        self.web_search_processor = WebSearchProcessor(config)
    
    def process_web_search_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        处理联网搜索结果并返回面向用户的回复文本。
        （原文：Processes web search results and returns a user-friendly response.）

        参数：
            query：用户当前提出的问题
            chat_history：最近几轮对话历史（每条为 {"role": ..., "content": ...}），可为 None
        返回：
            str：LLM 基于搜索结果生成的医学总结回复
        """
        return self.web_search_processor.process_web_results(query, chat_history)
