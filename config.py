
import os
from dotenv import load_dotenv

load_dotenv()

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default

def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default

class LLMSettings:

    def __init__(self):
        self.base_url = _env("LLM_BASE_URL")
        self.model = _env("LLM_MODEL", "gpt-4o")
        self.api_key = _env("OPENAI_API_KEY") or _env("LLM_API_KEY")
        self.fallback_base_url = _env("LLM_FALLBACK_BASE_URL")
        self.fallback_model = _env("LLM_FALLBACK_MODEL")
        self.fallback_api_key = _env("LLM_FALLBACK_API_KEY")
        self.request_timeout = _env_int("LLM_TIMEOUT_SEC", 60)
        self.max_retries = _env_int("LLM_MAX_RETRIES", 2)

class EmbeddingSettings:

    def __init__(self):
        self.provider = _env("EMBEDDING_PROVIDER", "local_bge_m3")
        self.model_name = _env("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.dim = _env_int("EMBEDDING_DIM", 1024)
        self.device = _env("EMBEDDING_DEVICE", "cpu")
        self.api_base_url = _env("EMBEDDING_API_BASE_URL")
        self.api_key = _env("EMBEDDING_API_KEY") or _env("OPENAI_API_KEY")

class RerankerSettings:

    def __init__(self):
        self.model_name = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.top_k = _env_int("RERANKER_TOP_K", 5)
        self.device = _env("RERANKER_DEVICE", "cpu")
        self.enabled = _env_bool("RERANKER_ENABLED", True)

class MilvusSettings:

    def __init__(self):
        self.uri = _env("MILVUS_URI")
        self.token = _env("MILVUS_TOKEN")
        self.local_path = _env("MILVUS_LOCAL_PATH", "./data/milvus_lite.db")
        self.collection = _env("MILVUS_COLLECTION", "mihc_knowledge")
        self.dim = _env_int("MILVUS_DIM", 1024)

class ElasticSettings:

    def __init__(self):
        self.url = _env("ES_URL")
        self.index = _env("ES_INDEX", "mihc_keywords")
        self.local_index_path = _env("ES_LOCAL_INDEX_PATH", "./data/bm25_index.json")

class PostgresSettings:

    def __init__(self):
        self.url = _env("DATABASE_URL", "sqlite:///./data/mihc.db")

class RedisSettings:

    def __init__(self):
        self.url = _env("REDIS_URL")
        self.use_fake = not self.url or _env_bool("USE_FAKE_REDIS", True)
        self.session_ttl = _env_int("SESSION_TTL_SEC", 3600)

class MongoSettings:

    def __init__(self):
        self.url = _env("MONGO_URL")
        self.db = _env("MONGO_DB", "mihc_platform")
        self.local_dir = _env("MONGO_LOCAL_DIR", "./data/mongo_json")

class IntentSettings:

    def __init__(self):
        self.model_path = _env("INTENT_MODEL_PATH")
        self.base_model = _env("INTENT_BASE_MODEL", "bert-base-chinese")
        self.labels = [
            "literature_search",
            "knowledge_qa",
            "data_analysis",
            "experiment_design",
            "other",
        ]
        self.min_confidence = _env_float("INTENT_MIN_CONFIDENCE", 0.6)
        self.device = _env("INTENT_DEVICE", "cpu")

class RetrievalSettings:

    def __init__(self):
        self.top_k = _env_int("RETRIEVAL_TOP_K", 5)
        self.keyword_top_k = _env_int("KEYWORD_TOP_K", 5)
        self.rrf_k = _env_int("RRF_K", 60)
        self.enable_rewrite = _env_bool("ENABLE_QUERY_REWRITE", True)
        self.enable_hyde = _env_bool("ENABLE_HYDE", True)
        self.min_retrieval_confidence = _env_float("MIN_RETRIEVAL_CONFIDENCE", 0.40)
        self.max_context_chars = _env_int("MAX_CONTEXT_CHARS", 8192)

class SecuritySettings:

    def __init__(self):
        self.dfa_words_file = _env("DFA_WORDS_FILE", "./security/sensitive_words.txt")
        self.enable_prompt_guard = _env_bool("ENABLE_PROMPT_GUARD", True)
        self.enable_input_scan = _env_bool("ENABLE_INPUT_SCAN", True)
        self.enable_output_scan = _env_bool("ENABLE_OUTPUT_SCAN", True)

class ObservabilitySettings:

    def __init__(self):
        self.otel_enabled = _env_bool("OTEL_ENABLED", True)
        self.otel_exporter = _env("OTEL_EXPORTER", "console")
        self.otel_endpoint = _env("OTEL_ENDPOINT", "http://localhost:4317")
        self.otel_service_name = _env("OTEL_SERVICE_NAME", "mihc-agent-platform")
        self.langfuse_public_key = _env("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = _env("LANGFUSE_SECRET_KEY")
        self.langfuse_host = _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.prometheus_enabled = _env_bool("PROMETHEUS_ENABLED", True)

class PlatformSettings:

    def __init__(self):
        self.name = "MIHC Agent"
        self.tenant_id = _env("TENANT_ID", "mihc")
        self.max_history = _env_int("MAX_CONVERSATION_HISTORY", 20)
        self.max_agent_steps = _env_int("MAX_AGENT_STEPS", 8)
        self.tool_timeout_sec = _env_int("TOOL_TIMEOUT_SEC", 30)
        self.fast_mode = _env_bool("FAST_MODE", False)

class MihcSettings:

    def __init__(self):
        self.intent_min_confidence = _env_float("MIHC_INTENT_MIN_CONFIDENCE", 0.75)
        self.object_storage_dir = _env("MIHC_OBJECT_STORAGE_DIR", "./data/objects")
        self.literature_dir = _env("MIHC_LITERATURE_DIR", "./data/literature")
        self.max_table_rows = _env_int("MIHC_MAX_TABLE_ROWS", 500000)
        self.max_project_files = _env_int("MIHC_MAX_PROJECT_FILES", 100)
        self.default_report_template = _env("MIHC_REPORT_TEMPLATE", "mihc_default_v1")

class AuthSettings:

    def __init__(self):
        self.secret = _env("MIHC_AUTH_SECRET", "dev-only-secret-change-me")
        self.token_ttl_sec = _env_int("MIHC_AUTH_TOKEN_TTL", 86400)
        self.allow_register = _env_bool("MIHC_ALLOW_REGISTER", True)
        self.admin_user = _env("MIHC_ADMIN_USER")
        self.admin_password = _env("MIHC_ADMIN_PASSWORD")

class APISettings:

    def __init__(self):
        self.host = _env("API_HOST", "0.0.0.0")
        self.port = _env_int("API_PORT", 8000)
        self.debug = _env_bool("API_DEBUG", True)
        self.max_upload_size_mb = _env_int("MAX_UPLOAD_SIZE_MB", 20)
        self.cors_origins = [origin.strip() for origin in _env("MIHC_CORS_ORIGINS", "*").split(",") if origin.strip()]

class Config:

    def __init__(self):
        self.llm = LLMSettings()
        self.embedding = EmbeddingSettings()
        self.reranker = RerankerSettings()
        self.milvus = MilvusSettings()
        self.elastic = ElasticSettings()
        self.postgres = PostgresSettings()
        self.redis = RedisSettings()
        self.mongo = MongoSettings()
        self.intent = IntentSettings()
        self.retrieval = RetrievalSettings()
        self.security = SecuritySettings()
        self.observability = ObservabilitySettings()
        self.platform = PlatformSettings()
        self.mihc = MihcSettings()
        self.auth = AuthSettings()
        self.api = APISettings()

    @property
    def llm_base_url(self) -> str:
        return self.llm.base_url

    @property
    def llm_api_key(self) -> str:
        return self.llm.api_key

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @property
    def llm_fallback_base_url(self) -> str:
        return self.llm.fallback_base_url

    @property
    def llm_fallback_model(self) -> str:
        return self.llm.fallback_model

    @property
    def llm_fallback_api_key(self) -> str:
        return self.llm.fallback_api_key

    @property
    def embedding_provider(self) -> str:
        return self.embedding.provider

    @property
    def embedding_model_name(self) -> str:
        return self.embedding.model_name

    @property
    def embedding_dim(self) -> int:
        return self.embedding.dim

    @property
    def embedding_device(self) -> str:
        return self.embedding.device

    @property
    def embedding_api_base_url(self) -> str:
        return self.embedding.api_base_url

    @property
    def embedding_api_key(self) -> str:
        return self.embedding.api_key
