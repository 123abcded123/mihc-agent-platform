"""
RAG 子模块组件：Reranker（cross-encoder 重排）。
角色：对向量检索返回的 top-k 文档做精排——
用 cross-encoder 计算查询与每篇文档的相关度分，
与原相似度分平均得到 combined_score=(余弦+重排)/2，
按分降序取前 top_k，并提取文档中引用的图片路径。
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model for more accurate results.

    中文说明：重排器。cross-encoder 把查询与文档拼接后整体编码打分，
    比双塔余弦相似度更精确；医疗场景可用 PubMedBERT 微调模型。
    """
    def __init__(self, config):
        """
        Initialize the reranker with configuration.

        中文说明：初始化重排器，加载 cross-encoder 模型并读取 top_k。
        
        Args:
            config: Configuration object containing reranker settings
                    配置对象，含 rag.reranker_model 与 rag.reranker_top_k
        """
        self.logger = logging.getLogger(__name__)
        
        # Load the cross-encoder model for reranking
        # For medical data, specialized models like 'pritamdeka/S-PubMedBert-MS-MARCO'
        # would be ideal, but using a general one here for simplicity
        # 加载 cross-encoder 模型：医疗数据理想上可用 PubMedBERT 微调模型，
        # 此处为简化使用通用重排模型
        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            # 模型加载失败时直接抛出，避免静默降级掩盖配置错误
            self.logger.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: Union[List[Dict[str, Any]], List[str]], parsed_content_dir: str) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using cross-encoder.

        中文说明：重排主流程：
        1) 兼容两种输入：字符串列表（补默认分）或字典列表（补缺失字段）；
        2) cross-encoder 打分 → combined_score=(原分+重排分)/2；
        3) 按 combined_score 降序取前 top_k；
        4) 提取块内容中的 picture_counter_N 标记，拼出图片引用 URL。
        重排失败时降级返回原排序，保证查询流程不中断。
        
        Args:
            query: User query
                   用户查询（已扩展）
            documents: Either a list of documents (dictionaries) or a list of strings
                       检索结果：字典列表或字符串列表
            parsed_content_dir: 解析产物目录，用于拼接图片 URL
            
        Returns:
            Reranked list of documents with updated scores
            重排后的文档列表（含 combined_score）与图片引用路径列表
        """
        try:
            if not documents:
                return []
            
            # Handle different document formats and ensure consistent structure
            # 统一文档格式：确保每个文档都有 id/content/score 字段
            if documents:
                # if the retrieved documents is just a list of strings, we add a default score
                # 字符串列表输入：转换为字典并给默认分 1.0
                if isinstance(documents[0], str):
                    # Convert simple strings to dictionaries
                    docs_list = []
                    for i, doc_text in enumerate(documents):
                        docs_list.append({
                            "id": i,
                            "content": doc_text,
                            "score": 1.0  # Default score
                        })
                    documents = docs_list
                # if the retrieved documents is a list of dictionaries, we use the original score
                # 字典列表输入：补齐缺失的 id/score/content 字段，保留原始分数
                elif isinstance(documents[0], dict):
                    # Ensure all required fields exist in dictionaries
                    for i, doc in enumerate(documents):
                        # Ensure ID exists
                        if "id" not in doc:
                            doc["id"] = i
                        # Ensure score exists
                        if "score" not in doc:
                            doc["score"] = 1.0
                        # Ensure content exists (unlikely to be missing but just in case)
                        # content 缺失时兼容 "text" 字段写法，否则用占位文本兜底
                        if "content" not in doc:
                            if "text" in doc:  # Some implementations might use "text" instead
                                doc["content"] = doc["text"]
                            else:
                                doc["content"] = f"Document {i}"
            
            # Create query-document pairs for scoring
            # 构造 (查询, 文档) 对，cross-encoder 逐对打分
            pairs = [(query, doc["content"]) for doc in documents]
            
            # Get relevance scores
            # cross-encoder 输出每对的相关度分数（越高越相关）
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            for i, score in enumerate(scores):
                # 记录纯重排分，供信心分计算或调试使用
                documents[i]["rerank_score"] = float(score)  # Store the new score from reranking
                # If the original document didn't have a score, use the rerank score
                # 原分缺失时补默认 1.0，保证后续平均计算不报错
                if "score" not in documents[i]:
                    documents[i]["score"] = 1.0
                # Combine (average) the original score and rerank score
                # 综合分 = (向量余弦分 + cross-encoder 重排分) / 2：
                # 融合检索阶段语义分与精排阶段相关度，两者互补更稳定
                documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
            
            # Sort by combined score
            # 按综合分降序排列
            reranked_docs = sorted(documents, key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to top_k if needed
            # 截断到 top_k，减少下游 LLM 上下文压力
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            
            # Extract picture references
            # 提取块内容中的 picture_counter_N 标记（由 content_processor 写入），
            # 反查对应图片文件并拼成可访问的静态服务 URL
            picture_reference_paths = []
            for doc in reranked_docs:
                matches = re.finditer(r"picture_counter_(\d+)", doc["content"])
                for match in matches:
                    counter_value = int(match.group(1))
                    # Create picture path based on document source and counter
                    # 图片名约定：<文档名>-picture-<序号>.png，由 doc_parser 导出时命名
                    doc_basename = os.path.splitext(doc['source'])[0]  # Remove file extension
                    # picture_path = Path(os.path.abspath(parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")).as_uri()
                    picture_path = os.path.join("http://localhost:8000/", parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")
                    picture_reference_paths.append(picture_path)
            
            return reranked_docs, picture_reference_paths
            
        except Exception as e:
            # 重排失败降级为原排序：宁可返回原始检索结果，也不让查询流程整体失败
            self.logger.error(f"Error during reranking: {e}")
            # Fallback to original ranking if reranking fails
            self.logger.warning("Falling back to original ranking")
            return documents
