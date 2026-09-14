"""
RAG 子模块组件：QueryExpander（查询扩展）。
角色：用 LLM 扩充用户查询中的医学术语、同义词与相关概念，
提升向量检索召回率；扩展后的查询贯穿后续检索与重排阶段。
"""
import logging
from typing import List, Dict, Any

class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.

    中文说明：查询扩展器。基于 LLM 的术语扩展，
    不改动用户原始意图，只补充同义词/医学术语以便匹配更多相关块。
    """
    def __init__(self, config):
        # 使用配置中的通用 LLM 做查询扩展
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand the original query with relevant medical terms.

        中文说明：对原始查询做术语扩展，返回原文与扩展文。
        若 LLM 判断无需扩展，扩展文保持原样。
        
        Args:
            original_query: The user's original query
                            用户的原始查询
            
        Returns:
            Dictionary with original and expanded queries
            字典：{"original_query": 原查询, "expanded_query": 扩展后查询}
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # Generate expansions - implement one of the strategies below
        # 调用 LLM 生成扩展查询
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with medical terminology."""
        # 查询扩展提示词：要求以医疗专家视角补充同义词/相关概念，
        # 无需扩展时保持原样；只输出扩展后的查询本身，不带任何解释
        prompt = f"""
        As a medical expert, expand the following query with relevant medical terminology, 
        synonyms, and related concepts that would help in retrieving relevant medical information:
        
        User Query: {query}
        
        Expand the query only if you feel like it is required, otherwise keep the user query intact.
        Be specific to the medical or any other domain mentioned in the ueer query, do not add other medical domains.
        If the user query asks about answering in tabular format, include that in the expanded query and do not answer in tabular format yourself.
        Provide only the expanded query without explanations.
        """
        expansion = self.model.invoke(prompt)
        
        return expansion
