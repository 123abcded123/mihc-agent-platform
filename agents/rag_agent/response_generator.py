"""
RAG 子模块组件：ResponseGenerator（回答生成）。
角色：基于重排后的文档生成最终回答——
只按资料内容作答（不臆造）、附来源链接与引用图片链接，
并计算信心分（前 3 块综合分的平均值）。
"""
import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.

    中文说明：回答生成器。组装系统提示词（表格格式指引+只依据上下文的约束），
    调用 LLM 生成回答，并从检索文档中提取去重来源与图片引用。
    """
    def __init__(self, config):
        """
        Initialize the response generator.

        中文说明：初始化回答生成器，读取生成模型实例与是否附来源的开关。
        
        Args:
            config: Configuration object
                    配置对象
            llm: Large language model for response generation
                 用于回答生成的大语言模型
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        # 是否在回答末尾附带来源文档链接（默认开启）
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Build the prompt for the language model.

        中文说明：构造回答提示词：
        拼接表格格式指引 + 检索上下文 + 回答格式约束（只依据上下文、不编造来源）。
        
        Args:
            query: User query
                   用户查询
            context: Formatted context from retrieved documents
                     检索文档拼接成的上下文
            chat_history: Optional chat history
                          可选的多轮对话历史
            
        Returns:
            Complete prompt string
            完整的提示词字符串
        """

        # 表格处理指引：要求检索到的表格数据用规范 markdown 表格呈现，
        # 并解释表格数据、不虚构表格内容
        table_instructions = """
        Some of the retrieved information is presented in table format. When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers, like this:
            | Column1 | Column2 | Column3 |
            |---------|---------|---------|
            | Value1  | Value2  | Value3  |
        2. Re-format the table structure to make it easier to read and understand
        3. If any new component is introduced during re-formatting of the table, mention it explicitly
        4. Clearly interpret the tabular data in your response
        5. Reference the relevant table when presenting specific data points
        6. If appropriate, summarize trends or patterns shown in the tables
        7. If only reference numbers are mentioned and you can fetch the corresponding values like research paper title or authors from the context, replace the reference numbers with the actual values
        """

        # 回答格式约束：只依据上下文作答、信息不足时明确说明、
        # 数值不得编造、禁止臆造来源链接
        response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, state: "I don't have enough information to answer this question based on the provided context."
        3. Do not use prior knowledge not contained in the context.
        5. Be concise and accurate.
        6. Provide a well-structured response with heading, sub-headings and tabular structure if required in markdown format based on retrieved knowledge. Keep the headings and sub-headings small sized.
        7. Only provide sections that are meaningful to have in a chatbot reply. For example, do not explicitly mention references.
        8. If values are involved, make sure to respond with perfect values present in context. Do not make up values.
        9. Do not repeat the question in the answer or response."""
            
        # Build the prompt
        # 总提示词：角色设定 + 对话历史 + 用户问题 + 检索上下文 + 两组指引，
        # 强调只依据资料回答、不编造来源，是 RAG 防幻觉的核心约束
        prompt = f"""You are a medical assistant providing accurate information based on verified medical sources.

        Here are the last few messages from our conversation:
        
        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        {table_instructions}

        {response_format_instructions}

        Based on the provided information, please answer the user's question thoroughly but concisely.
        If the information doesn't contain the answer, acknowledge the limitations of the available information.

        Do not provide any source link that is not present in the context. Do not make up any source link.

        Medical Assistant Response:"""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
        """
        Generate a response based on retrieved documents.

        中文说明：回答生成主流程：
        1) 拼接检索文档为上下文并构建提示词；
        2) LLM 生成回答；
        3) 提取去重来源链接附在回答末尾；
        4) 附加引用图片链接；
        5) 计算信心分（前 3 块综合分平均）。
        
        Args:
            query: User query
                   用户查询
            retrieved_docs: List of retrieved document dictionaries
                            重排后的检索文档列表
            picture_paths: 引用图片的 URL 列表
            chat_history: Optional chat history
                          可选的多轮对话历史
            
        Returns:
            Dict containing response text and source information
            字典：{"response": 回答文本, "sources": 来源列表, "confidence": 信心分}
        """
        try:
           
            # Extract content from documents for context
            # 取出每个文档的正文内容
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # Combine retrieved documents into a single context
            # 用分隔线拼接成单一上下文，让 LLM 能区分不同来源块
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # Build the prompt
            # 组装完整提示词
            prompt = self._build_prompt(query, context, chat_history)
            
            # Generate response
            # 调用生成模型产出回答
            response = self.response_generator_model.invoke(prompt)
            
            # Extract sources for citation
            # 提取去重后的来源链接（开关开启时才提取）
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Calculate confidence
            # 计算信心分：取前 3 块综合分（或无综合分时的原始分）平均
            confidence = self._calculate_confidence(retrieved_docs)

            # Add sources to response
            # 来源以 markdown 链接形式追加到回答末尾，方便用户回溯原始资料
            if hasattr(self, 'include_sources') and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content
            
            # Add picture paths to response
            # 引用的图片也以链接形式追加，展示相关图表
            response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
            
            # Format final response
            # 组装最终返回结构
            result = {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            # 生成失败时返回兜底话术，避免向上抛异常中断对话服务
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents for citation.

        中文说明：从检索文档提取来源信息：
        按 (source, source_path) 去重，按综合分（降级到重排分/原始分）排序，
        最终只输出 title 与 path（分数仅用于排序）。
        
        Args:
            documents: List of retrieved document dictionaries
                       检索文档字典列表
            
        Returns:
            List of source information dictionaries
            去重并按相关性排序的来源列表：[{"title":..., "path":...}]
        """
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in documents:
            # Extract source and source_path
            # 提取来源文件名与链接
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # Skip if no source information is available
            # 无来源信息的文档直接跳过，不参与引用
            if not source:
                continue
                
            # Create a unique identifier for this source
            # 用 (source, source_path) 组合作为唯一键去重
            source_id = f"{source}|{source_path}"
            
            # Skip if we've already included this source
            # 同一来源（不同块）只保留一次
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
            # 分数取值优先级：综合分 > 重排分 > 原始检索分
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        # 按相关度降序，保证最重要的来源排在最前
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        # 去掉仅供排序使用的 score 字段，输出干净的 title/path 结构
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.

        中文说明：计算信心分——前 3 块综合分的平均值。
        分数取值优先级：combined_score（重排后）> rerank_score > score（余弦）。
        无文档时返回 0.0。
        
        Args:
            documents: Retrieved documents
                       检索（重排后）文档
            
        Returns:
            Confidence score between 0 and 1
            0~1 之间的信心分
        """
        if not documents:
            return 0.0
            
        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        # 优先用综合分（余弦+重排的平均），其次重排分，最后原始余弦分
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        # 前 3 块（不足 3 块时取全部）的平均分作为最终信心分
        return sum(scores) / len(scores) if scores else 0.0
