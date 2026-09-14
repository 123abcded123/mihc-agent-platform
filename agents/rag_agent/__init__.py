"""
RAG 子模块总编排（MedicalRAG）。
角色：作为多智能体医疗助手中的检索增强生成（RAG）中枢，
负责协调文档入库管线（解析→图片摘要→格式化→语义切块→建库）
与查询管线（查询扩展→混合检索→重排→生成回答）。
本文件被上层 agent 调用，本身不实现具体算法，只做流程串联与日志/统计。
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any

# 导入 RAG 子模块各组件：文档解析、内容处理、向量库、重排、查询扩展、回答生成
from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system that integrates all components.

    中文说明：医疗检索增强生成系统（MedicalRAG），
    负责把 doc_parser / content_processor / vector_store / reranker /
    query_expander / response_generator 六大组件整合成完整 RAG 管线。
    """
    def __init__(self, config):
        """
        Initialize the RAG Agent.

        中文说明：初始化 RAG Agent，创建全部子组件并读取配置。
        
        Args:
            config: Configuration object with RAG settings
                    配置对象，包含 RAG 相关的模型名、路径、top_k 等设置
        """
        # Set up logging
        # 配置日志记录器，用于追踪入库与查询全流程
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        # 实例化六大组件：解析器不依赖配置，其余组件均从 config.rag 读取参数
        self.doc_parser = MedicalDocParser()
        self.content_processor = ContentProcessor(config)
        self.vector_store = VectorStore(config)
        self.reranker = Reranker(config)
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        # 解析产物（页面/图片 PNG）的落地目录，供图片引用路径拼接使用
        self.parsed_content_dir = self.config.rag.parsed_content_dir
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all files in a directory into the RAG system.

        中文说明：批量入库——遍历目录下所有文件，逐个调用 ingest_file，
        汇总成功/失败统计信息。失败文件不会中断整个批量流程。
        
        Args:
            directory_path: Path to the directory containing files to ingest
                            待入库文件所在的目录路径
            
        Returns:
            Dictionary with ingestion results
            入库统计字典：成功文档数、失败数、处理块数、耗时等
        """
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            # Check if directory exists
            # 目录不存在时直接抛错，避免后续 os.listdir 报底层异常
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Get all files in the directory
            # 仅收集目录下的常规文件（忽略子目录），拼接完整路径
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                # 空目录按成功处理，但文档数与块数均为 0
                self.logger.warning(f"No files found in directory: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Track statistics
            # 统计变量：累计块数、成功/失败文件数、失败文件明细
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            
            # Process each file
            # 逐个处理文件；单个文件异常被捕获，不中断其余文件的入库
            for file_path in files:
                self.logger.info(f"Processing file {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # 目录级异常（如目录不存在）统一返回失败结果
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the RAG system.

        中文说明：单文件入库管线（5 步）：
        parse_document（PDF→结构化文档+图片）→ summarize_images（LLM 看图摘要）
        → format_document_with_images（摘要替换图片占位符）→ chunk_document（语义切块）
        → create_vectorstore（双向量写入 Qdrant + 原文写入 docstore）。
        
        Args:
            document_path: Path to the file to ingest
                            待入库的文档文件路径
            
        Returns:
            Dictionary with ingestion results
            入库结果字典：成功标志、文档数、切块数、耗时
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # Step 1: Parse document
            # 步骤1：PDF 解析为结构化文档，并提取图片保存到 parsed_content_dir
            self.logger.info("1. Parsing document and extracting images...")
            parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            self.logger.info(f"   Parsed document and extracted {len(images)} images")

            # Step 2: Summarize images
            # 步骤2：用 LLM 逐张生成图片摘要，为后续替换占位符做准备
            self.logger.info("2. Summarizing images...")
            image_summaries = self.content_processor.summarize_images(images)
            self.logger.info(f"   Generated {len(image_summaries)} image summaries")

            # Step 3: Format document with image summaries
            # 步骤3：把 markdown 中的图片占位符替换为对应摘要文本
            self.logger.info("3. Formatting document with image summaries...")
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)

            # Step 4: Chunk document into semantic sections
            # 步骤4：LLM 语义切块（每块 256-512 词），保证主题一致
            self.logger.info("4. Chunking document into semantic sections...")
            document_chunks = self.content_processor.chunk_document(formatted_document)
            self.logger.info(f"   Document split into {len(document_chunks)} chunks")

            # Step 5: Create vector store and document store
            # 步骤5：块向量（dense+sparse）写入 Qdrant，块原文写入 docstore
            self.logger.info("5. Creating vector store knowledge base...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
                )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            # 单文件入库失败时返回错误信息，由上层（目录批量）汇总
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query with the RAG system.

        中文说明：查询管线（4 步）：
        expand_query（LLM 扩充医学术语）→ load_vectorstore + retrieve_relevant_chunks
        （hybrid 检索 top5）→ rerank（cross-encoder 打分取前3）→ generate_response
        （只按资料回答、附来源链接、算信心分）。
        
        Args:
            query: The query string
                    用户原始查询
            chat_history: Optional chat history for context
                    可选的多轮对话历史，用于生成回答时提供上下文
            
        Returns:
            Response dictionary
            包含回答文本、来源、信心分与耗时的字典
        """
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # Process query and return result, passing chat_history
        # 查询处理主流程；任何一步异常都捕获并返回兜底错误回答
        try:
            # Step 1: Expand query
            # 步骤1：用 LLM 扩充查询中的医学术语/同义词，提升召回率
            self.logger.info(f"1. Expanding query: '{query}'")
            expansion_result = self.query_expander.expand_query(query)
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Original: '{query}'")
            self.logger.info(f"   Expanded: '{expanded_query}'")
            query = expanded_query

            # Step 2: Retrieval
            # 步骤2：加载向量库，hybrid（dense+sparse）检索 top-k 相关块
            self.logger.info(f"2. Retrieving relevant documents for the query: '{query}'")
            vectorstore, docstore = self.vector_store.load_vectorstore()
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                docstore=docstore,
                )

            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks")

            # Step 3: Rerank the retrieved documents if we have a reranker and enough documents
            # 步骤3：文档数>1 且重排器可用时，用 cross-encoder 精排；
            # 否则降级为原始检索顺序（保证流程不中断）
            self.logger.info(f"3. Reranking the retrieved documents")
            if self.reranker and len(retrieved_documents) > 1:
                reranked_documents, reranked_top_k_picture_paths = self.reranker.rerank(query, retrieved_documents, self.parsed_content_dir)
                self.logger.info(f"   Reranked retrieved documents and chose top {len(reranked_documents)}")
                self.logger.info(f"   Found {len(reranked_top_k_picture_paths)} referenced images")
            else:
                self.logger.info(f"   Could not rerank the retrieved documents, falling back to original scores")
                reranked_documents = retrieved_documents
                reranked_top_k_picture_paths = []

            # Step 4: Generate response
            # 步骤4：基于重排后的文档生成回答，附来源链接并计算信心分
            self.logger.info("4. Generating response...")
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=reranked_documents,
                picture_paths=reranked_top_k_picture_paths,
                chat_history=chat_history
                )
            
            # Add timing information
            # 附加上总耗时，便于监控与统计
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            
            return response
        
        except Exception as e:
            # 查询失败时记录完整堆栈并返回兜底回答，避免向上抛出导致服务中断
            self.logger.error(f"Error processing query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
