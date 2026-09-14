"""
tavily_search.py：Tavily 通用网络搜索引擎封装。

基于 langchain_community 的 TavilySearchResults 工具实现，
每次检索最多返回 5 条结果，并把 title/url/content/score 格式化为纯文本。
"""
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

class TavilySearchAgent:
    """
    Tavily 搜索引擎封装类。

    原英文注释：
    Processes general documents for the RAG system with context-aware chunking.
    """
    def __init__(self):
        """
        初始化 Tavily 搜索代理。
        （原文：Initialize the Tavily search agent.）
        
        参数（Args）：
            query: User query
        """
        pass

    def search_tavily(self, query: str) -> str:
        """Perform a general web search using Tavily API.
        执行 Tavily 通用网络搜索：最多取 5 条结果，格式化为 "title - url - content - score" 的纯文本。

        参数：
            query：搜索查询语句（会先去除首尾引号）
        返回：
            str：多行格式化结果文本；无结果时返回 "No relevant results found."；
                 出错时返回 "Error retrieving web search results: <异常信息>"
        """

        # 创建 Tavily 搜索工具，限制最多返回 5 条结果
        tavily_search = TavilySearchResults(max_results = 5)

        # 以下为曾经使用的原生 requests 调用方式，现已改用 langchain 工具，故注释保留
        # url = "https://api.tavily.com/search"
        # params = {
        #     "api_key": tavily_api_key,
        #     "query": query,
        #     "num_results": 5
        # }
        
        try:
            # response = requests.get(url, params=params)
            # Strip any surrounding quotes from the query
            # 去除查询语句首尾可能存在的引号
            query = query.strip('"\'')
            # print("Printing query:", query)
            # 调用 Tavily 工具执行搜索，返回结构化文档列表
            search_docs = tavily_search.invoke(query)
            # data = response.json()
            # if "results" in data:
            # 若存在搜索结果，则逐条格式化为 title/url/content/score 文本
            if len(search_docs):
                return "\n".join(["title: " + str(res["title"]) + " - " + 
                                  "url: " + str(res["url"]) + " - " + 
                                  "content: " + str(res["content"]) + " - " + 
                                  "score: " + str(res["score"]) for res in search_docs])
            # 无结果时给出友好提示
            return "No relevant results found."
        except Exception as e:
            # 任何异常都转为错误文本返回，不让异常向上传播
            return f"Error retrieving web search results: {e}"
