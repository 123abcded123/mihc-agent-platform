"""
MIHC 医疗科研智能 Agent 平台 —— 集中配置文件

本文件在系统中的角色：
1. 单一配置源：LLM（vLLM Qwen3-32B / OpenAI 兼容）、嵌入（BGE-M3）、
   检索（Milvus + ES/BM25 + BGE-Reranker）、存储（PostgreSQL / Redis / MongoDB）、
   安全（DFA + Guardrail）、观测（OTel / Langfuse / Prometheus）等参数集中于此；
2. 环境变量入口：通过 python-dotenv 从 .env 加载全部外部服务连接信息；
3. 多后端适配：所有基础设施组件均支持"生产实现 + 本地回退"两种模式，
   无 Docker/无服务器的开发机也能完整运行（详见《重构方案.md》第 2 节）。
"""

import os
from dotenv import load_dotenv

# 加载 .env 中的环境变量
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
    """LLM 配置：vLLM(Qwen3-32B LoRA) 为主，OpenAI 为回退（模型网关故障切换）。"""

    def __init__(self):
        # 主端点：vLLM 服务地址（http://host:port/v1）；留空则直接用 OpenAI 官方端点
        self.base_url = _env("LLM_BASE_URL")
        # 主模型名：vLLM 部署时填服务名（如 qwen3-32b-lora），OpenAI 时填 gpt-4o
        self.model = _env("LLM_MODEL", "gpt-4o")
        self.api_key = _env("OPENAI_API_KEY") or _env("LLM_API_KEY")
        # 备用端点（模型网关故障切换）
        self.fallback_base_url = _env("LLM_FALLBACK_BASE_URL")
        self.fallback_model = _env("LLM_FALLBACK_MODEL")
        self.fallback_api_key = _env("LLM_FALLBACK_API_KEY")
        # 调用约束
        self.request_timeout = _env_int("LLM_TIMEOUT_SEC", 60)
        self.max_retries = _env_int("LLM_MAX_RETRIES", 2)


class EmbeddingSettings:
    """嵌入配置：BGE-M3 本地推理 / OpenAI 兼容嵌入端点。"""

    def __init__(self):
        self.provider = _env("EMBEDDING_PROVIDER", "local_bge_m3")  # local_bge_m3 | openai_compatible
        self.model_name = _env("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.dim = _env_int("EMBEDDING_DIM", 1024)  # BGE-M3 稠密维度
        self.device = _env("EMBEDDING_DEVICE", "cpu")  # 本地推理设备（4GB 显存建议 cpu）
        self.api_base_url = _env("EMBEDDING_API_BASE_URL")
        self.api_key = _env("EMBEDDING_API_KEY") or _env("OPENAI_API_KEY")


class RerankerSettings:
    """重排配置：BGE-Reranker（CrossEncoder）；云端无模型文件时可关闭并降级为得分排序。"""

    def __init__(self):
        self.model_name = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.top_k = _env_int("RERANKER_TOP_K", 5)
        self.device = _env("RERANKER_DEVICE", "cpu")
        self.enabled = _env_bool("RERANKER_ENABLED", True)


class MilvusSettings:
    """Milvus 向量库：milvus-lite 本地文件模式 / Milvus Server。"""

    def __init__(self):
        # 留空 → milvus-lite 本地模式（data/milvus_lite.db）；填 URI → 连接服务端
        self.uri = _env("MILVUS_URI")
        self.token = _env("MILVUS_TOKEN")
        self.local_path = _env("MILVUS_LOCAL_PATH", "./data/milvus_lite.db")
        self.collection = _env("MILVUS_COLLECTION", "mihc_knowledge")
        self.dim = _env_int("MILVUS_DIM", 1024)


class ElasticSettings:
    """Elasticsearch 关键词检索：真实 ES / 本地 BM25 回退。"""

    def __init__(self):
        self.url = _env("ES_URL")
        self.index = _env("ES_INDEX", "mihc_keywords")
        # 本地 BM25 回退索引文件（ES_URL 为空时使用）
        self.local_index_path = _env("ES_LOCAL_INDEX_PATH", "./data/bm25_index.json")


class PostgresSettings:
    """PostgreSQL（SQLAlchemy）：文档元数据/版本/权限/审计；无服务时回退 SQLite。"""

    def __init__(self):
        self.url = _env("DATABASE_URL", "sqlite:///./data/mihc.db")


class RedisSettings:
    """Redis 会话上下文/任务进度；无服务时回退 fakeredis。"""

    def __init__(self):
        self.url = _env("REDIS_URL")
        self.use_fake = not self.url or _env_bool("USE_FAKE_REDIS", True)
        self.session_ttl = _env_int("SESSION_TTL_SEC", 3600)


class MongoSettings:
    """MongoDB：badcase / 评测数据；无服务时回退 JSON 文件存储。"""

    def __init__(self):
        self.url = _env("MONGO_URL")
        self.db = _env("MONGO_DB", "mihc_platform")
        self.local_dir = _env("MONGO_LOCAL_DIR", "./data/mongo_json")


class IntentSettings:
    """BERT 意图识别配置（5 个业务意图 + other）。"""

    def __init__(self):
        # 微调后的 BERT 权重路径；为空时直接用预训练 bert-base-chinese（需 fine-tune 才能达到生产准确率）
        self.model_path = _env("INTENT_MODEL_PATH")
        self.base_model = _env("INTENT_BASE_MODEL", "bert-base-chinese")
        self.labels = [
            "literature_search",   # 文献检索
            "knowledge_qa",        # 知识问答
            "lab_interpretation",  # 检验解读
            "data_analysis",       # 数据分析
            "experiment_design",   # 实验设计
            "other",               # 其他（闲聊/非科研问题）
        ]
        self.min_confidence = _env_float("INTENT_MIN_CONFIDENCE", 0.6)
        self.device = _env("INTENT_DEVICE", "cpu")


class RetrievalSettings:
    """检索链参数（Query Rewrite / HyDE / RRF / Top-K / 置信度）。"""

    def __init__(self):
        self.top_k = _env_int("RETRIEVAL_TOP_K", 5)          # 向量路召回数
        self.keyword_top_k = _env_int("KEYWORD_TOP_K", 5)    # 关键词路召回数
        self.rrf_k = _env_int("RRF_K", 60)                   # RRF 融合常数
        self.enable_rewrite = _env_bool("ENABLE_QUERY_REWRITE", True)
        self.enable_hyde = _env_bool("ENABLE_HYDE", True)
        self.min_retrieval_confidence = _env_float("MIN_RETRIEVAL_CONFIDENCE", 0.40)
        self.max_context_chars = _env_int("MAX_CONTEXT_CHARS", 8192)


class SecuritySettings:
    """安全配置：DFA 敏感词 + Prompt 护栏 + 审计。"""

    def __init__(self):
        self.dfa_words_file = _env("DFA_WORDS_FILE", "./security/sensitive_words.txt")
        self.enable_prompt_guard = _env_bool("ENABLE_PROMPT_GUARD", True)
        self.enable_input_scan = _env_bool("ENABLE_INPUT_SCAN", True)
        self.enable_output_scan = _env_bool("ENABLE_OUTPUT_SCAN", True)


class ObservabilitySettings:
    """观测配置：OpenTelemetry / Langfuse / Prometheus。"""

    def __init__(self):
        self.otel_enabled = _env_bool("OTEL_ENABLED", True)
        self.otel_exporter = _env("OTEL_EXPORTER", "console")  # console | otlp
        self.otel_endpoint = _env("OTEL_ENDPOINT", "http://localhost:4317")
        self.otel_service_name = _env("OTEL_SERVICE_NAME", "mihc-agent-platform")
        # Langfuse：三个 key 都配置才启用
        self.langfuse_public_key = _env("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = _env("LANGFUSE_SECRET_KEY")
        self.langfuse_host = _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.prometheus_enabled = _env_bool("PROMETHEUS_ENABLED", True)


class PlatformSettings:
    """平台标识与多租户。"""

    def __init__(self):
        self.name = "MIHC Agent"
        self.tenant_id = _env("TENANT_ID", "mihc")
        self.max_history = _env_int("MAX_CONVERSATION_HISTORY", 20)
        self.max_agent_steps = _env_int("MAX_AGENT_STEPS", 8)   # Agent 最大步数（防死循环）
        self.tool_timeout_sec = _env_int("TOOL_TIMEOUT_SEC", 30)
        # 快速模式：LLM 端点延迟高时启用——
        # 规则意图 + 单步规划 + 仅 DFA 护栏 + 关闭改写/HyDE/LLM 引用审查，
        # 每个请求只保留 1 次 LLM 生成调用
        self.fast_mode = _env_bool("FAST_MODE", False)


class MihcSettings:
    """客户 mIHC 科研服务配置。"""

    def __init__(self):
        # 低于该置信度时只澄清，不直接调用业务 Agent。
        self.intent_min_confidence = _env_float("MIHC_INTENT_MIN_CONFIDENCE", 0.75)
        self.object_storage_dir = _env("MIHC_OBJECT_STORAGE_DIR", "./data/objects")
        self.max_table_rows = _env_int("MIHC_MAX_TABLE_ROWS", 500000)
        self.max_project_files = _env_int("MIHC_MAX_PROJECT_FILES", 100)
        self.default_report_template = _env("MIHC_REPORT_TEMPLATE", "mihc_default_v1")


class AuthSettings:
    """登录鉴权配置（HMAC 签名 token，无外部依赖）。"""

    def __init__(self):
        self.secret = _env("MIHC_AUTH_SECRET", "dev-only-secret-change-me")
        self.token_ttl_sec = _env_int("MIHC_AUTH_TOKEN_TTL", 86400)
        self.allow_register = _env_bool("MIHC_ALLOW_REGISTER", True)
        self.admin_user = _env("MIHC_ADMIN_USER")
        self.admin_password = _env("MIHC_ADMIN_PASSWORD")


class APISettings:
    """FastAPI 服务参数。"""

    def __init__(self):
        self.host = _env("API_HOST", "0.0.0.0")
        self.port = _env_int("API_PORT", 8000)
        self.debug = _env_bool("API_DEBUG", True)
        self.max_upload_size_mb = _env_int("MAX_UPLOAD_SIZE_MB", 20)
        # 允许跨域的前端来源，逗号分隔；默认 *（Bearer token 鉴权，不依赖 Cookie）
        self.cors_origins = [origin.strip() for origin in _env("MIHC_CORS_ORIGINS", "*").split(",") if origin.strip()]


class Config:
    """全局配置聚合类（门面）。

    业务代码只需 `from config import Config` 实例化一次即可访问全部配置。
    """

    def __init__(self):
        self.llm = LLMSettings()                    # LLM 模型网关
        self.embedding = EmbeddingSettings()        # BGE-M3 嵌入
        self.reranker = RerankerSettings()          # BGE-Reranker
        self.milvus = MilvusSettings()              # 向量库
        self.elastic = ElasticSettings()            # ES/BM25 关键词库
        self.postgres = PostgresSettings()          # 关系型元数据/审计
        self.redis = RedisSettings()                # 会话上下文
        self.mongo = MongoSettings()                # badcase/评测
        self.intent = IntentSettings()               # BERT 意图
        self.retrieval = RetrievalSettings()         # 检索链参数
        self.security = SecuritySettings()           # DFA + 护栏
        self.observability = ObservabilitySettings() # OTel/Langfuse/Prometheus
        self.platform = PlatformSettings()           # 平台/租户
        self.mihc = MihcSettings()                    # 客户 mIHC 科研服务
        self.auth = AuthSettings()                    # 登录鉴权
        self.api = APISettings()                     # API 服务

    # ---- 便捷属性（兼容旧模块/减少重复取值）----
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
