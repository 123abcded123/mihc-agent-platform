"""
web_search_agent.py：搜索引擎门面（Facade）。

统一封装各种搜索引擎，对外只暴露 search(query) 一个入口。
当前启用 Tavily 通用搜索；PubMed 学术搜索已被注释停用（保留接口以备后续启用）。
"""
import requests
from typing import Dict

from .pubmed_search import PubmedSearchAgent
from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    """
    搜索引擎门面：负责从网络来源检索实时医学信息。

    原英文注释：
    Agent responsible for retrieving real-time medical information from web sources.
    """
    
    def __init__(self, config):
        # 初始化 Tavily 搜索引擎（当前唯一启用的引擎）
        self.tavily_search_agent = TavilySearchAgent()
        
        # PubMed 学术搜索目前被停用，相关初始化代码已注释保留
        # self.pubmed_search_agent = PubmedSearchAgent()
        # self.pubmed_api_url = config.pubmed_api_url
    
    def search(self, query: str) -> str:
        """
        执行联网检索并拼接各引擎的结果。

        当前只调用 Tavily（返回 title/url/content/score 格式化文本）；
        PubMed 的调用已被注释停用。

        参数：
            query：经过 LLM 压缩后的搜索查询语句
        返回：
            str：以 "Tavily Results:" 为标题的格式化搜索结果文本
            （原文：Perform both general and medical-specific searches.）
        """
        # print(f"[WebSearchAgent] Searching for: {query}")
        
        # 调用 Tavily 引擎执行检索
        tavily_results = self.tavily_search_agent.search_tavily(query=query)
        # PubMed 检索目前被停用
        # pubmed_results = self.pubmed_search_agent.search_pubmed(self.pubmed_api_url, query)
        
        # 拼接并返回 Tavily 结果文本（PubMed 部分已被注释掉）
        return f"Tavily Results:\n{tavily_results}\n"
        # \nPubMed Results:\n{pubmed_results}"
