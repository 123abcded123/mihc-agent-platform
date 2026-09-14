"""
pubmed_search.py：PubMed 生物医学文献检索封装（当前已被停用）。

通过 NCBI E-utilities API 检索 PubMed 文献，返回最多 5 篇相关文章的链接。
注意：WebSearchAgent 中对该类的调用已被注释，此模块目前不参与实际流程。
"""
import requests

class PubmedSearchAgent:
    """
    PubMed 检索封装类（目前已停用，仅保留备用）。

    原英文注释：
    Processes medical documents for the RAG system with context-aware chunking.
    """
    def __init__(self):
        """
        初始化 Pubmed 搜索代理。
        （原文：Initialize the Pubmed search agent.）
        
        参数（Args）：
            query: User query
        """
        pass

    def search_pubmed(self, pubmed_api_url, query: str) -> str:
        """Search PubMed for relevant medical articles.
        检索 PubMed 医学文献：调用 E-utilities esearch 接口，返回最多 5 篇文章的链接。

        参数：
            pubmed_api_url：PubMed E-utilities API 地址（由外部配置传入）
            query：检索关键词
        返回：
            str：多行文章链接文本；无结果时返回 "No relevant PubMed articles found."；
                 出错时返回 "Error retrieving PubMed articles: <异常信息>"
        """
        # 构造 esearch 请求参数：数据库为 pubmed，JSON 返回，最多 5 条
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5
        }
        
        try:
            # 调用 E-utilities 接口检索文献 ID 列表
            response = requests.get(pubmed_api_url, params=params)
            data = response.json()
            article_ids = data.get("esearchresult", {}).get("idlist", [])
            # 没有命中任何文献时给出提示
            if not article_ids:
                return "No relevant PubMed articles found."
            
            # 把文献 ID 拼成 PubMed 详情页链接
            article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/" for article_id in article_ids]
            return "\n".join(article_links)
        except Exception as e:
            # 请求失败时返回错误文本
            return f"Error retrieving PubMed articles: {e}"
