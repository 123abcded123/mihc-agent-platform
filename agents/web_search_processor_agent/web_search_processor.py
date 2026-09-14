"""
web_search_processor.py：联网搜索处理核心流程。

完整数据流（三段式）：
1. 用 LLM 把“用户当前问题 + 最近对话历史”压缩成一条适合搜索引擎的查询语句；
2. 调用 WebSearchAgent（底层为 Tavily，max_results=5）执行联网检索；
3. 把原始搜索结果交给 LLM，生成一份医学准确、简洁友好的总结回复。

注意：LLM 实例来自 config.web_search.llm，与主对话 LLM 分开配置。
"""
import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如 Tavily API Key 等）
load_dotenv()

class WebSearchProcessor:
    """
    联网搜索处理器：串联“LLM 压缩查询 → 搜索 → LLM 总结”整条链路。

    原英文注释：
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        # 创建搜索引擎门面（内部封装 Tavily 等搜索引擎）
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        # 获取专门用于“压缩查询 / 总结结果”的 LLM 实例（配置来自 config.web_search.llm）
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        构造“查询压缩”提示词。

        让 LLM 结合最近对话历史，把用户当前问题重写为一条独立、完整、
        适合直接用于联网搜索的查询语句（仅当历史与当前问题相关时才合并）。
        
        参数：
            query：用户当前提出的问题
            chat_history：最近几轮对话历史，可为 None
            
        返回：
            str：完整的提示词文本

        原文：
        Build the prompt for the web search.

        Args:
            query: User query
            chat_history: chat history

        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # （预留）如提供对话历史，会一并放入提示词供 LLM 参考
        # print("Chat History:", chat_history)
            
        # Build the prompt
        # 组装提示词模板：最近对话 + 当前问题 + 压缩指令
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        联网搜索主流程（对外入口）。

        流程：
        1. 构造查询压缩提示词并调用 LLM，得到适合搜索的查询语句；
        2. 用该查询语句调用搜索引擎（Tavily，最多 5 条结果）；
        3. 把搜索结果拼进总结提示词，再调用 LLM 生成医学总结。

        参数：
            query：用户原始问题
            chat_history：最近对话历史，可为 None
        返回：
            AIMessage：LLM 生成的最终回复（调用方可用 .content 取出文本）
            （原文：Fetches web search results, processes them using LLM, and returns a user-friendly response.）
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        # 第一步：构建“查询压缩”提示词
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        
        # 第二步：调用 LLM 压缩出适合搜索的查询语句
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)

        # Retrieve web search results
        # 第三步：用压缩后的查询语句执行联网检索（Tavily，最多 5 条结果）
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results
        # 第四步：组装“总结”提示词，要求 LLM 基于可靠来源生成医学准确的简洁回复
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
        
        # Invoke the LLM to process the results
        # 第五步：调用 LLM 汇总搜索结果，生成最终回复
        response = self.llm.invoke(llm_prompt)
        
        return response
