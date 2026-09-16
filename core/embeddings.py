
from __future__ import annotations

import logging
from typing import List

from core.errors import MIHCError

logger = logging.getLogger(__name__)

class BaseEmbedder:
    dim: int = 1024

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

class LocalBGE_M3Embedder(BaseEmbedder):

    LOCAL_PATH = "./models/bge-m3"

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        import os
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MIHCError("本地嵌入需要 sentence-transformers；云端部署请配置 EMBEDDING_API_BASE_URL "
                            "并设置 EMBEDDING_PROVIDER=openai_compatible") from exc
        if os.path.isdir(self.LOCAL_PATH):
            model_name = self.LOCAL_PATH
        logger.info("Loading local embedding model %s on %s ...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

class OpenAICompatibleEmbedder(BaseEmbedder):

    def __init__(self, base_url: str | None, api_key: str | None, model: str, dim: int = 1024):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.dim = dim

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        try:
            resp = self.client.embeddings.create(model=self.model, input=inputs)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            raise MIHCError(f"嵌入服务调用失败: {exc}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

class EmbedderFactory:

    def __init__(self, config):
        self.config = config
        self._embedder: BaseEmbedder | None = None

    def get(self) -> BaseEmbedder:
        if self._embedder is None:
            provider = self.config.embedding_provider
            if provider == "openai_compatible":
                self._embedder = OpenAICompatibleEmbedder(
                    base_url=self.config.embedding_api_base_url,
                    api_key=self.config.embedding_api_key,
                    model=self.config.embedding_model_name,
                    dim=self.config.embedding_dim,
                )
            else:
                self._embedder = LocalBGE_M3Embedder(
                    model_name=self.config.embedding_model_name,
                    device=self.config.embedding_device,
                )
            logger.info("Embedding backend: %s (dim=%s)", provider, self._embedder.dim)
        return self._embedder
