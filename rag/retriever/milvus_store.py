
from __future__ import annotations

import logging
import uuid
from typing import List, Dict, Any, Optional

from core.errors import RetrievalError

logger = logging.getLogger(__name__)

class MilvusStore:
    def __init__(self, config):
        self.config = config
        self.collection_name = config.milvus.collection
        self.dim = config.milvus.dim
        self.uri = config.milvus.uri
        self.local_path = config.milvus.local_path
        self._client = None
        self._loaded: set = set()

    def _ensure_client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            if self.uri:
                self._client = MilvusClient(uri=self.uri, token=self.config.milvus.token or None)
                backend = self.uri
            else:
                self._client = MilvusClient(self.local_path)
                backend = f"milvus-lite:{self.local_path}"
            logger.info("Milvus backend: %s", backend)
        return self._client

    def _ensure_collection(self, load: bool = True):
        client = self._ensure_client()
        if not client.has_collection(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
                id_type="string",
                primary_field_name="id",
                vector_field_name="embedding",
            )
            logger.info("Milvus collection created: %s", self.collection_name)
        if load and self.collection_name not in self._loaded:
            try:
                client.load_collection(self.collection_name)
                self._loaded.add(self.collection_name)
                logger.info("Milvus collection loaded: %s", self.collection_name)
            except Exception as exc:
                logger.warning("Milvus load_collection failed (may already be loaded): %s", exc)

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0
        self._ensure_collection()
        data = []
        for chunk, vec in zip(chunks, embeddings):
            data.append({
                "id": chunk["chunk_id"],
                "text": chunk.get("text", ""),
                "title": chunk.get("title", ""),
                "source": chunk.get("source", ""),
                "version": chunk.get("version", ""),
                "tenant_id": chunk.get("tenant_id", "mihc"),
                "doc_id": chunk.get("doc_id", ""),
                "embedding": vec,
            })
        self._ensure_client().upsert(collection_name=self.collection_name, data=data)
        return len(data)

    def search(self, query_embedding: List[float], top_k: int = 5,
               tenant_id: str = "mihc", allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self._ensure_collection()
        client = self._ensure_client()
        expr_parts = [f'tenant_id == "{tenant_id}"']
        if allowed_doc_ids:
            ids = ", ".join(f'"{d}"' for d in allowed_doc_ids)
            expr_parts.append(f"doc_id in [{ids}]")
        try:
            results = client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=" and ".join(expr_parts),
                search_params={"metric_type": "COSINE"},
                output_fields=["text", "title", "source", "version", "doc_id"],
            )
        except Exception as exc:
            raise RetrievalError(f"Milvus 检索失败: {exc}")

        out = []
        for hit in (results[0] if results else []):
            entity = hit.get("entity", {})
            out.append({
                "chunk_id": entity.get("id", ""),
                "text": entity.get("text", ""),
                "title": entity.get("title", ""),
                "source": entity.get("source", ""),
                "version": entity.get("version", ""),
                "doc_id": entity.get("doc_id", ""),
                "score": float(hit.get("distance", 0.0)),
            })
        return out

    def delete_by_doc(self, doc_id: str) -> None:
        self._ensure_collection()
        self._ensure_client().delete(collection_name=self.collection_name,
                                     filter=f'doc_id == "{doc_id}"')

    def count(self) -> int:
        self._ensure_collection()
        stats = self._ensure_client().get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))
