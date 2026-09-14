"""
RAG 子模块组件：VectorStore（Qdrant 向量库读写）。
角色：管理 Qdrant 集合的创建/加载与文档入库/检索。
collection 采用双向量：dense（1536 维余弦）+ sparse（BM25 关键词）；
检索模式为 RetrievalMode.HYBRID（混合检索，语义+关键词互补）。
docstore 为 LocalFileStore，key=doc_id，value=块原文 UTF-8 字节，
供检索时回取父文档原文。
"""
import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStore:
    """
    Create vector store, ingest documents, retrieve relevant documents

    中文说明：向量库管理器。封装 Qdrant 本地持久化客户端，
    负责创建双向量 collection、入库文档（向量+原文双写）、
    加载已有库并执行 hybrid 检索。
    """
    def __init__(self, config):
        # 从配置读取集合名、维度、距离度量、嵌入模型、top_k 及本地路径
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path

        # Use the singleton client instead of creating a new one
        # self.client = QdrantClientManager.get_client(config)
        # 使用本地文件路径模式创建 Qdrant 客户端（无需启动独立服务器，数据落在 vectorstore_local_path）
        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        # 检查集合是否存在：列出全部集合名并比对目标集合名
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            # 查询失败按不存在处理，调用方会尝试新建
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection with dense and sparse vectors."""
        # 创建双向量集合：dense 用余弦距离（语义向量），sparse 用 BM25 索引（关键词向量）
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        Load existing vectorstore and docstore for retrieval operations without ingesting new documents.

        中文说明：查询前加载已有向量库与文档库。
        集合不存在时直接抛错（提示先入库），不自动新建空库。
        
        Returns:
            Tuple containing (vectorstore, docstore)
            元组：(QdrantVectorStore 实例, LocalFileStore 实例)
        """
        # Check if collection exists
        # 先校验集合存在，避免空库检索产生误导性空结果
        if not self._does_collection_exist():
            self.logger.error(f"Collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Collection {self.collection_name} does not exist")
            
        # Setup sparse embeddings
        # 加载 BM25 稀疏嵌入器，与入库时保持一致
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Initialize vector store
        # 以 HYBRID 模式初始化：检索时同时利用 dense（语义）与 sparse（关键词）向量
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage
        # 加载 docstore（本地文件存储），用于按 doc_id 回取块原文
        docstore = LocalFileStore(self.docstore_local_path)
        
        self.logger.info(f"Successfully loaded existing vectorstore and docstore")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vector store from document chunks or upsert documents to existing store.

        中文说明：入库入口。双写策略：
        1) 每个块生成 uuid 作为 doc_id，写入 Qdrant（dense+sparse 双向量）；
        2) 同一 doc_id 作为 key 把块原文（UTF-8 字节）写入 docstore，
           检索命中后用 doc_id 回 docstore 取父文档原文。
        集合不存在则新建，存在则直接 upsert 追加。
        
        Args:
            document_chunks: List of document chunks
                             待入库的语义块列表
            document_path: Path to the original document
                           原始文档路径（用于元数据 source/source_path）
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
            元组：(向量库, 文档库, 块 ID 列表)
        """
        
        # Generate unique IDs for each chunk
        # 为每个块生成全局唯一 ID，作为 Qdrant 向量与 docstore 原文的关联键
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents
        # 构造 langchain Document：page_content 为块文本，metadata 记录来源与 doc_id
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        # "source_path": Path(os.path.abspath(document_path)).as_uri()
                        # 来源链接指向本地静态文件服务，供前端渲染引用
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        # Setup sparse embeddings
        # 加载 BM25 稀疏嵌入器，入库时同时计算关键词向量
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Check if collection exists, create if it doesn't
        # 集合不存在则创建；已存在则走 upsert 追加，不重建集合
        collection_exists = self._does_collection_exist()
        if not collection_exists:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
        else:
            self.logger.info(f"Collection {self.collection_name} already exists, will upsert documents")
        
        # Initialize vector store
        # 与 load_vectorstore 相同配置：HYBRID 模式 + dense/sparse 双向量
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage for parent documents
        # 打开 docstore，用于保存父文档（块）原文
        docstore = LocalFileStore(self.docstore_local_path)
        
        # Ingest documents into vector and doc stores
        # 双写：向量+元数据进 Qdrant，原文进 docstore
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        
        # Encode string chunks to bytes before storing
        # docstore 只接受字节值，先把块原文编码为 UTF-8 字节再批量写入
        encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: LocalFileStore,
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant chunks based on a query.

        中文说明：hybrid 检索 top-k。流程：
        1) similarity_search_with_score 做混合检索（语义+关键词互补），取 top_k；
        2) 用命中块的 doc_id 回 docstore 取父文档原文；
        3) 组装成 reranker 期望的字典结构（id/content/score/source/source_path）。
        
        Args:
            query: User query
                   用户查询（通常已经过查询扩展）
            vectorstore: Vector store containing embeddings
                         包含块向量的 Qdrant 向量库
            docstore: Document store containing actual content
                      保存块原文的文档库
            
        Returns:
            Tuple containing (retrieved_docs, picture_reference_paths)
            where retrieved_docs is a list of dictionaries with content and score
            检索结果列表（每项为含 id/content/score/source/source_path 的字典）
        """
        # Use similarity_search_with_score to get documents and scores
        # 混合检索：Qdrant 内部融合 dense 语义分与 sparse 关键词分，返回 top_k 命中及分数
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        # picture_reference_paths = []
        
        for chunk, score in results:
            # Get full document from doc store as bytes and decode to string
            # 用 doc_id 回 docstore 取父文档原文（字节），解码为 UTF-8 字符串
            doc_content_bytes = docstore.mget([chunk.metadata['doc_id']])[0]
            doc_content = doc_content_bytes.decode('utf-8')
            
            # Add metadata to the document
            # formatted_doc = f"{doc_content}\nFollowing are the 'filename' and 'path as uri' of the source document for the current chunk: {chunk.metadata['source']}, {chunk.metadata['source_path']}"
            # 此处直接使用原文内容；文件名与路径由独立字段承载
            formatted_doc = doc_content
            
            # Create document dict in the format expected by reranker
            # 组装重排器期望的字典结构：id/content/score/source/source_path
            doc_dict = {
                "id": chunk.metadata['doc_id'],
                "content": formatted_doc,
                "score": score,  # Use the actual similarity score
                "source": chunk.metadata['source'],
                "source_path": chunk.metadata['source_path'],
            }
            retrieved_docs.append(doc_dict)
            
            # # Extract picture references
            # matches = re.finditer(r"picture_counter_(\d+)", doc_content)
            # for match in matches:
            #     counter_value = int(match.group(1))
            #     # Create picture path based on document source and counter
            #     doc_basename = os.path.splitext(chunk.metadata['source'])[0]  # Remove file extension
            #     picture_path = Path(os.path.abspath(parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")).as_uri()
            #     picture_reference_paths.append(picture_path)
        
        # return retrieved_docs, picture_reference_paths
        # 图片引用提取已移至 reranker 中执行，此处只返回检索结果
        return retrieved_docs
