"""
配置文件（v4.1 — 接入 DeepSeek + 依赖注入 + 流式输出版）
支持 DEV / PROD / TEST 三套配置，通过 .env 文件切换
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)


# ===== v5.1: 自定义配置异常 =====
class ConfigurationError(Exception):
    """生产环境配置缺失或无效时抛出，替代 SystemExit。"""

    pass


# ===== v4.1: 环境标识（启动时输出）=====
_DEV_MODE = os.getenv("DEV_MODE", "").lower() == "true"
_ENV_LABEL = "DEV" if _DEV_MODE else "PROD"
logging.getLogger("config").info(f"Environment: {_ENV_LABEL} (DEV_MODE={_DEV_MODE})")


def _int_env(key: str, default: int) -> int:
    """v3.8 fix: safe int env conversion (prevents ValueError on invalid input)"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _float_env(key: str, default: float) -> float:
    """v3.8 fix: safe float env conversion"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# ===== OpenAI 兼容 API 配置（v4.1: 默认硅基流动）=====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "siliconflow")  # siliconflow | deepseek | openai | custom
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")  # v6.0: 从 Qwen2.5-7B 升级
LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 4096)

# ===== 系统配置（放在 HTTP 配置之前，因为 HTTP_HEADERS 引用 VERSION）=====
VERSION = os.getenv("APP_VERSION", "6.3")  # 可从环境变量覆盖，便于 CI/CD
APP_NAME = os.getenv("APP_NAME", "药妆智多星 - Customer Service AI Agent")
DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "面向化妆品企业的多智能体客服系统（企业级增强版）",
)

# ===== HTTP 请求配置 =====
HTTP_TIMEOUT = _int_env("HTTP_TIMEOUT", 15)  # v5.4: 从 30s 降至 15s，减少失败场景等待
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": f"MultiAgentCustomerService/{VERSION}",
}

# ===== 路由配置 =====
ROUTING_COMPLEXITY_THRESHOLD = _int_env("ROUTING_COMPLEXITY_THRESHOLD", 50)

# ===== v6.1: 缓存架构重构 =====
# L1: Redis 精确缓存
# （复用 REDIS_URL，不用新增变量）

# L2: Qdrant 语义缓存
CACHE_QDRANT_COLLECTION = os.getenv("CACHE_QDRANT_COLLECTION", "response_cache")
CACHE_QDRANT_MAX_POINTS = _int_env("CACHE_QDRANT_MAX_POINTS", 10000)
CACHE_VECTOR_SCORE_THRESHOLD = _float_env("CACHE_VECTOR_SCORE_THRESHOLD", 0.85)

# TTL 策略: intent_type → TTL(秒)
CACHE_TTL_POLICY = {
    "knowledge_qa": 604800,  # 7 天
    "pricing_stock": 300,  # 5 分钟
    "policy_rule": 86400,  # 24 小时
    "order_status": 300,  # 5 分钟
    "after_sales": 3600,  # 1 小时
    "chitchat": 600,  # 10 分钟
    "default": 3600,  # 1 小时
}

# L3: Jaccard 降级
CACHE_FALLBACK_ENABLED = os.getenv("CACHE_FALLBACK_ENABLED", "true").lower() == "true"
CACHE_FALLBACK_THRESHOLD = _float_env("CACHE_FALLBACK_THRESHOLD", 0.6)

# P0-02: 缓存内容版本。当知识库/提示词/产品目录发生大规模变更需要整体失效
# 旧缓存时，提升该版本即可使三层旧条目因版本不匹配而无法命中（无需显式迁移）。
# 注意：提升版本会使所有现有缓存条目失效（命中率短暂下降），属预期行为。
CACHE_CONTENT_VERSION = os.getenv("CACHE_CONTENT_VERSION", "1")

# 后台清理
CACHE_CLEANUP_INTERVAL = _int_env("CACHE_CLEANUP_INTERVAL", 3600)  # 秒

# 向后兼容别名（v6.1: 旧 env var 仍会读取，未设置时回退到新值）
CACHE_L1_MAX = _int_env("CACHE_L1_MAX", 500)  # 不再用于内存 L1，保留避免 ImportError
CACHE_L2_MAX = _int_env("CACHE_L2_MAX", 2000)  # 同上
CACHE_TTL = _int_env("CACHE_TTL", 3600)  # 仅作为 default_ttl 兼容
CACHE_SEMANTIC_THRESHOLD_SHORT = _float_env(
    "CACHE_SEMANTIC_THRESHOLD_SHORT", CACHE_VECTOR_SCORE_THRESHOLD
)
CACHE_SEMANTIC_THRESHOLD_LONG = _float_env(
    "CACHE_SEMANTIC_THRESHOLD_LONG", CACHE_VECTOR_SCORE_THRESHOLD
)

# ===== 会话配置 =====
SESSION_WINDOW_SIZE = _int_env("SESSION_WINDOW_SIZE", 10)
SESSION_STORAGE_BACKEND = os.getenv("SESSION_STORAGE_BACKEND", "memory")
SESSION_MAX_TOKENS = _int_env("SESSION_MAX_TOKENS", 4000)
SESSION_SUMMARY_MAX_CHARS = _int_env("SESSION_SUMMARY_MAX_CHARS", 500)

# ===== 漂移检测配置 =====
DRIFT_TOPIC_JACCARD_THRESHOLD = _float_env("DRIFT_TOPIC_JACCARD_THRESHOLD", 0.15)
DRIFT_REPETITION_THRESHOLD = _float_env("DRIFT_REPETITION_THRESHOLD", 0.8)
DRIFT_ESCALATION_THRESHOLD = _int_env("DRIFT_ESCALATION_THRESHOLD", 5)

# ===== 连接池配置（httpx） =====
HTTPX_MAX_CONNECTIONS = _int_env("HTTPX_MAX_CONNECTIONS", 100)
HTTPX_KEEPALIVE_CONNECTIONS = _int_env("HTTPX_KEEPALIVE_CONNECTIONS", 20)

# ===== 响应时长 SLA 配置（v3.1 补充） =====
RESPONSE_TIME_TARGET_MIN = _float_env("RESPONSE_TIME_TARGET_MIN", 5.0)
RESPONSE_TIME_TARGET_MAX = _float_env("RESPONSE_TIME_TARGET_MAX", 20.0)

# ===== SLA 告警配置（v3.2 新增）=====
SLA_ALERT_WINDOW = _int_env("SLA_ALERT_WINDOW", 50)
SLA_ALERT_THRESHOLD = _float_env("SLA_ALERT_THRESHOLD", 30.0)
SLA_ALERT_COOLDOWN = _int_env("SLA_ALERT_COOLDOWN", 300)

# ===== 模型熔断器配置（v3.2 新增）=====
CIRCUIT_BREAKER_FAIL_THRESHOLD = _int_env("CIRCUIT_BREAKER_FAIL_THRESHOLD", 5)
CIRCUIT_BREAKER_RECOVERY_TIME = _int_env("CIRCUIT_BREAKER_RECOVERY_TIME", 60)
LLM_ROUTER_TIMEOUT = _float_env(
    "LLM_ROUTER_TIMEOUT", 4.0
)  # v4.3: 从 8s 降至 4s，配合熔断器快速 fallback

# ===== 重试配置 =====
RETRY_MAX_ATTEMPTS = _int_env("RETRY_MAX_ATTEMPTS", 3)  # v6.1: 3→指数退避可达 1s/2s/4s 三级
RETRY_BASE_DELAY = _float_env("RETRY_BASE_DELAY", 1.0)
RETRY_MAX_DELAY = _float_env("RETRY_MAX_DELAY", 30.0)  # v6.1: 指数退避最大延迟
RETRY_BACKOFF_FACTOR = _int_env("RETRY_BACKOFF_FACTOR", 2)  # v6.1: 指数退避因子

# ===== v3.4: 安全配置 ======
MAX_QUERY_LENGTH = _int_env("MAX_QUERY_LENGTH", 2000)
MAX_SESSIONS = _int_env("MAX_SESSIONS", 10000)
SESSION_IDLE_TTL = _int_env("SESSION_IDLE_TTL", 3600)

# ===== v3.7: 安全加固配置 =====
# 监控端点管理 Token（/api/metrics, /api/kpi 等敏感端点需要此 Token）
MONITORING_ADMIN_TOKEN = os.getenv("MONITORING_ADMIN_TOKEN", "")
# WebSocket 连接限制
WS_MAX_CONNECTIONS_PER_IP = _int_env("WS_MAX_CONNECTIONS_PER_IP", 5)
WS_MESSAGE_RATE_LIMIT = _int_env("WS_MESSAGE_RATE_LIMIT", 10)
WS_IDLE_TIMEOUT = _int_env("WS_IDLE_TIMEOUT", 300)
# 会话令牌签名密钥（用于防会话劫持）
SESSION_TOKEN_SECRET = os.getenv("SESSION_TOKEN_SECRET", "")
# 会话数据加密密钥（用于文件/Redis 后端存储加密；为空则明文存储，向后兼容）
SESSION_ENCRYPTION_KEY = os.getenv("SESSION_ENCRYPTION_KEY", "")
# TLS 配置
TLS_CERT_FILE = os.getenv("TLS_CERT_FILE", "")  # TLS 证书文件路径
TLS_KEY_FILE = os.getenv("TLS_KEY_FILE", "")  # TLS 私钥文件路径

# ===== v3.8: Token Quota 配置 =====
TOKEN_QUOTA_DAILY = _int_env("TOKEN_QUOTA_DAILY", 100000)  # 每日 Token 上限
TOKEN_QUOTA_MONTHLY = _int_env("TOKEN_QUOTA_MONTHLY", 2000000)  # 每月 Token 上限
TOKEN_QUOTA_ENABLED = os.getenv("TOKEN_QUOTA_ENABLED", "true").lower() == "true"  # 是否启用 Quota
TOKEN_QUOTA_REDIS_PREFIX = os.getenv("TOKEN_QUOTA_REDIS_PREFIX", "csai:quota:")  # Redis key 前缀


# ===== v6.1: 限流配置（供组件注册表引用）=====
RATE_LIMIT_MAX = _int_env("RATE_LIMIT_MAX", 60)  # 每分钟最大请求数
RATE_LIMIT_WINDOW = _int_env("RATE_LIMIT_WINDOW", 60)  # 时间窗口（秒）


# ===== v6.1: 基准测试 / 证据缺口修复配置 =====
BENCHMARK_REPORT_DIR = os.getenv("BENCHMARK_REPORT_DIR", "reports")
BENCHMARK_QUERY_COUNT = _int_env("BENCHMARK_QUERY_COUNT", 500)  # 默认评测集大小
BENCHMARK_SEED = _int_env("BENCHMARK_SEED", 42)  # 可复现随机种子
BENCHMARK_CACHE_WARMUP = _int_env("BENCHMARK_CACHE_WARMUP", 200)  # 缓存预热查询数
BENCHMARK_TOP_K = _int_env("BENCHMARK_TOP_K", 3)  # RAG 评估 Top-K
BENCHMARK_PREFETCH_ENABLED = os.getenv("BENCHMARK_PREFETCH_ENABLED", "true").lower() == "true"


# ===== 日志配置 =====
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# ===== CORS 配置（v3.0 新增） =====
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# ===== API 认证配置（v3.0 新增） =====
# ===== 进程角色（config validation boundary） =====
#
# core/config.py 被**所有**进程导入：API 服务与 Celery worker。两者对配置的需求不同：
#   - API    : 需要 API Key 认证、CORS、端用户 JWT / Session 密钥；
#   - worker : 只执行 AgentRun，不终结 HTTP 请求、不签发或校验用户凭据，因此
#              API_KEY / CORS / JWT / SESSION_TOKEN_SECRET 对它没有意义。
#
# 过去 API 专属校验（尤其是下面这个 **import 期** 的硬 raise）会直接阻塞 worker 启动：
# worker 只要 import core.config 就必须提供 API_KEY，否则 ValueError 起不来，而 worker
# 根本不会读取这个配置。现在用 SERVICE_ROLE 显式区分，默认 `api` —— 生产 API 的行为
# 完全不变；只有显式声明 SERVICE_ROLE=worker 的进程才跳过 API 专属校验。
SERVICE_ROLE = os.getenv("SERVICE_ROLE", "api").strip().lower() or "api"
if SERVICE_ROLE not in ("api", "worker"):
    raise ConfigurationError(
        f"SERVICE_ROLE 非法: {SERVICE_ROLE!r}（仅支持 api | worker）"
    )
#: 当前进程是否为 Celery worker（据此跳过 API 专属校验）。
IS_WORKER_PROCESS = SERVICE_ROLE == "worker"

API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "true").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
# 仅 API 进程需要 API Key；worker 不终结 HTTP 请求，不应因缺 API_KEY 而无法启动。
if API_KEY_ENABLED and not API_KEY and not IS_WORKER_PROCESS:
    raise ValueError(
        "API_KEY_ENABLED=true requires a non-empty API_KEY. Set API_KEY in .env, "
        "disable authentication with API_KEY_ENABLED=false, or start this process with "
        "SERVICE_ROLE=worker if it is a Celery worker"
    )

# ===== ERP 模式配置（v3.0 新增） =====
ERP_MODE = os.getenv("ERP_MODE", "mock")  # mock | real
ERP_BASE_URL = os.getenv("ERP_BASE_URL", "")
ERP_APP_ID = os.getenv("ERP_APP_ID", "")
ERP_APP_SECRET = os.getenv("ERP_APP_SECRET", "")
ERP_DB_ID = os.getenv("ERP_DB_ID", "")

# P0-03 Remaining Risk #4: 权威 user_id → customer_id 映射（真实 ERP 模式）。
# 生产环境由认证系统 / ERP 客户绑定提供；此处通过环境变量配置 JSON 映射，
# 例如 ERP_USER_CUSTOMER_MAP='{"user_001":"C001","user_002":"C002"}'。
# 映射必须来自服务端可信数据，绝不来自 prompt / LLM / 资源自声明。
# 缺失 / 无效 / 非法时返回空 dict -> ErpAuthorizationService fail closed。
ERP_USER_CUSTOMER_MAP_RAW = os.getenv("ERP_USER_CUSTOMER_MAP", "")


def load_erp_user_customer_map(raw: str = ERP_USER_CUSTOMER_MAP_RAW) -> dict[str, str]:
    """解析 ERP_USER_CUSTOMER_MAP JSON 为 {user_id: customer_id} 映射。

    任何输入异常（空、非法 JSON、非 dict）一律返回空 dict（fail closed），
    绝不抛异常影响启动。值非字符串时强制转 str；null 值跳过。
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


ERP_USER_CUSTOMER_MAP: dict[str, str] = load_erp_user_customer_map()

# ===== Redis 配置（v3.0 新增） =====
_redis_url_env = os.getenv("REDIS_URL", "").strip()
if _redis_url_env:
    REDIS_URL = _redis_url_env
else:
    _redis_password = os.getenv("REDIS_PASSWORD", "")
    _redis_host = os.getenv("REDIS_HOST", "localhost")
    REDIS_URL = (
        f"redis://:{_redis_password}@{_redis_host}:6379"
        if _redis_password
        else f"redis://{_redis_host}:6379"
    )

# ===== v3.5: RAG 配置 =====
RAG_PERSIST_DIRECTORY = os.getenv("RAG_PERSIST_DIRECTORY", "")  # 空则内存模式
RAG_N_RESULTS = _int_env("RAG_N_RESULTS", 3)
RAG_QUERY_REWRITING = (
    os.getenv("RAG_QUERY_REWRITING", "false").lower() == "true"
)  # v5.2: LLM 改写查询

# ===== v6.2: Embedding & Reranker API 配置（替代本地 sentence-transformers）=====
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
# v6.3: 当 EMBEDDING_API_KEY 未单独配置时，回退到 LLM 的 OPENAI_API_KEY
if not EMBEDDING_API_KEY:
    EMBEDDING_API_KEY = OPENAI_API_KEY
    logging.getLogger("config").info(
        "EMBEDDING_API_KEY 未配置，将复用 OPENAI_API_KEY。"
        "生产环境建议配置独立的 Embedding 服务 API Key"
    )
EMBEDDING_DIM = _int_env("EMBEDDING_DIM", 1024)  # bge-large-zh-v1.5 输出维度

RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")

# ===== Hybrid retrieval (vector + BM25) configuration =====
HYBRID_SEARCH_ENABLED = (
    os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
)  # 混合检索总开关
HYBRID_RRF_K = _int_env("HYBRID_RRF_K", 60)  # RRF rank 平滑常数
HYBRID_VECTOR_TOP_K = _int_env("HYBRID_VECTOR_TOP_K", 8)  # 向量通道每 collection top-K
HYBRID_BM25_TOP_K = _int_env("HYBRID_BM25_TOP_K", 8)  # BM25 通道每 collection top-K

# P1-02: BM25 lifecycle configuration.
BM25_REBUILD_TIMEOUT = _int_env("BM25_REBUILD_TIMEOUT", 60)
BM25_COLLECTIONS = ["product_knowledge", "faq", "tech_support", "complaint_knowledge"]

# ===== v6.0: Qdrant 向量数据库配置 =====
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = _int_env("QDRANT_PORT", 6333)  # REST API 端口
QDRANT_GRPC_PORT = _int_env("QDRANT_GRPC_PORT", 6334)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
QDRANT_COLLECTION_CONFIG = {
    "vectors": {"size": 1024, "distance": "Cosine"},
    "optimizers_config": {"default_segment_number": 2},
    "hnsw_config": {"m": 16, "ef_construct": 100},
}
# v6.2: 仅使用 Qdrant
VECTOR_DB_MODE = os.getenv("VECTOR_DB_MODE", "qdrant_only")

# ===== v3.5: ReAct 配置 =====
REACT_MAX_ITERATIONS = _int_env("REACT_MAX_ITERATIONS", 3)  # v4.3: 从 5 降至 3，控制延迟在 20s 内
REACT_COMPLEXITY_THRESHOLD = _int_env("REACT_COMPLEXITY_THRESHOLD", 60)
REACT_SELF_REFLECTION = os.getenv("REACT_SELF_REFLECTION", "false").lower() == "true"

# ===== v3.5: 工具调用配置 =====
TOOL_MAX_ROUNDS = _int_env("TOOL_MAX_ROUNDS", 3)

# Agent Context Engineering: deterministic Tool Result Context Budget.
TOOL_RESULT_OPTIMIZATION_ENABLED = (
    os.getenv("TOOL_RESULT_OPTIMIZATION_ENABLED", "false").lower() == "true"
)
TOOL_RESULT_MAX_TOKENS = _int_env("TOOL_RESULT_MAX_TOKENS", 800)
TOOL_RESULT_MAX_ITEMS = _int_env("TOOL_RESULT_MAX_ITEMS", 5)
TOOL_RESULT_PRESERVE_RECENT = _int_env("TOOL_RESULT_PRESERVE_RECENT", 2)
TOOL_RESULT_OFFLOAD_ENABLED = os.getenv("TOOL_RESULT_OFFLOAD_ENABLED", "false").lower() == "true"
TOOL_RESULT_OFFLOAD_MIN_TOKENS = _int_env("TOOL_RESULT_OFFLOAD_MIN_TOKENS", 1200)
TOOL_RESULT_STORE_TTL_SECONDS = _int_env("TOOL_RESULT_STORE_TTL_SECONDS", 900)
TOOL_RESULT_SEMANTIC_SUMMARY_ENABLED = (
    os.getenv("TOOL_RESULT_SEMANTIC_SUMMARY_ENABLED", "false").lower() == "true"
)
TOOL_RESULT_CACHE_ENABLED = os.getenv("TOOL_RESULT_CACHE_ENABLED", "false").lower() == "true"

# ===== MCP（Model Context Protocol）外部工具接入 =====
# 默认**关闭**。MCP 工具是来自外部进程（可能是第三方）的不可信输入源，
# 与进程内 native 工具的信任级别不同，因此必须显式开启 + 显式 allowlist。
#
# MCP_SERVERS 是 JSON 数组 allowlist，每条形如：
#   {"name": "catalog", "transport": "stdio", "command": "python3",
#    "args": ["-m", "my_mcp_server"], "allowed_tools": ["get_product_info"],
#    "timeout_seconds": 15.0, "max_payload_bytes": 32768,
#    "enabled": true, "risk_level": "low"}
#
# ``risk_level`` 与 ``core.hitl.risk.RiskLevel`` **同源**，只能取
# ``low`` / ``medium`` / ``high``（见 MCP_RISK_LEVELS），不另立词汇表。
#
# **read-only-first**：只有显式声明 ``"low"`` 的 server 的工具才会被注册
# （见 tools/mcp_adapter.py::register_mcp_tools）。缺失或非法的 ``risk_level``
# 一律收敛到 ``HIGH``（只向上收敛，绝不向下），因此其工具**不会**进入
# Function Calling 表面。本模块尚未为写操作接通幂等 ledger 与人工审批链路，
# 接入写操作 MCP 工具必须先补这两道防线。
MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
MCP_SERVERS = os.getenv("MCP_SERVERS", "")
MCP_DEFAULT_TIMEOUT_SECONDS = _float_env("MCP_DEFAULT_TIMEOUT_SECONDS", 15.0)
MCP_MAX_PAYLOAD_BYTES = _int_env("MCP_MAX_PAYLOAD_BYTES", 32768)
# 初始化失败时是否 fail closed（抛出，阻止服务启动）还是降级为 native-only。
# 生产默认应设 true：静默降级会让「以为 MCP 已生效」的运维假设与实际不符。
MCP_FAIL_CLOSED = os.getenv("MCP_FAIL_CLOSED", "false").lower() == "true"

# ===== v4.0: 用户认证配置 =====
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_HOURS = _int_env("JWT_EXPIRE_HOURS", 72)
JWT_ACCESS_EXPIRE_HOURS = _int_env("JWT_ACCESS_EXPIRE_HOURS", 2)  # P2-3: access_token 短生命周期
JWT_REFRESH_EXPIRE_HOURS = _int_env("JWT_REFRESH_EXPIRE_HOURS", 168)  # P2-3: refresh_token 7天
# v4.0: 本地开发模式（仅开发环境设置此变量为 true，生产环境禁止）
DEV_MODE = os.getenv("DEV_MODE", "").lower() == "true"

# ===== v4.0: 数据库配置 =====
DB_DIR = os.getenv("DB_DIR", "data")
DB_PATH = os.getenv("DB_PATH", os.path.join(DB_DIR, "csai.db"))

# ===== v4.0: 告警通知配置 =====
ALERT_WEBHOOKS = os.getenv(
    "ALERT_WEBHOOKS", ""
)  # JSON 数组: [{"name":"钉钉","url":"...","type":"dingtalk"}]
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = _int_env("SMTP_PORT", 587)
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# ===== v4.1: 数据库配置（PostgreSQL + SQLite 兼容）=====
DATABASE_URL = os.getenv("DATABASE_URL", "")  # 为空则使用 SQLite
ALEMBIC_CONFIG_PATH = os.getenv("ALEMBIC_CONFIG_PATH", "alembic.ini")

# ===== LangGraph Checkpoint 持久化配置 =====
# 背景：MemorySaver 只存在于当前 Python 进程内，gunicorn 多 worker / 多副本
# 之间不共享，实例重启即丢失。生产需要官方 PostgreSQL checkpointer。
#   memory   -> langgraph.checkpoint.memory.MemorySaver（进程内，仅开发/测试）
#   postgres -> langgraph.checkpoint.postgres.AsyncPostgresSaver（跨进程/重启持久化）
# 留空表示按环境自动选择：开发/测试->memory，生产->postgres。
LANGGRAPH_CHECKPOINT_BACKEND_RAW = os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "").strip().lower()
# 显式 checkpoint 数据库 URL。留空且 backend=postgres 时安全复用 DATABASE_URL
# （仅接受 libpq 兼容的 postgresql:// / postgres://，剥离 SQLAlchemy +driver 后缀；
#  SQLite / 其它协议一律拒绝，绝不静默降级）。
LANGGRAPH_CHECKPOINT_DATABASE_URL = os.getenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", "").strip()
# psycopg 连接池大小与首次 setup 超时（秒）
LANGGRAPH_CHECKPOINT_POOL_MIN_SIZE = _int_env("LANGGRAPH_CHECKPOINT_POOL_MIN_SIZE", 1)
LANGGRAPH_CHECKPOINT_POOL_MAX_SIZE = _int_env("LANGGRAPH_CHECKPOINT_POOL_MAX_SIZE", 10)
LANGGRAPH_CHECKPOINT_SETUP_TIMEOUT = _float_env("LANGGRAPH_CHECKPOINT_SETUP_TIMEOUT", 15.0)


def resolve_checkpoint_backend(raw: str = "", *, dev_mode: bool = False) -> str:
    """解析 checkpoint backend，非法值 fail closed。

    空值按环境选择（开发/测试 memory，生产 postgres）；显式值只允许
    ``memory`` / ``postgres``，其它值抛 ``ConfigurationError``。
    """
    value = (raw or "").strip().lower()
    if not value:
        return "memory" if dev_mode else "postgres"
    if value not in ("memory", "postgres"):
        raise ConfigurationError(
            f"LANGGRAPH_CHECKPOINT_BACKEND 非法: {value!r}（仅支持 memory | postgres）"
        )
    return value


def derive_checkpoint_database_url(explicit: str = "", *, database_url: str = "") -> str | None:
    """把 SQLAlchemy 风格 DSN 安全转换为 psycopg 可用的 libpq DSN。

    只接受 postgres 协议；剥离 ``postgresql+psycopg2://`` 这类 SQLAlchemy
    driver 后缀，标准化 ``postgres://`` -> ``postgresql://``。SQLite / MySQL /
    无 scheme 的输入返回 ``None``（由调用方 fail closed），不做字符串乱替换。
    """
    raw = (explicit or "").strip() or (database_url or "").strip()
    if not raw:
        return None
    scheme, sep, rest = raw.partition("://")
    if not sep:
        return None
    base = scheme.split("+", 1)[0].lower()
    if base in ("postgres", "postgresql"):
        return f"postgresql://{rest}"
    return None


LANGGRAPH_CHECKPOINT_BACKEND = resolve_checkpoint_backend(
    LANGGRAPH_CHECKPOINT_BACKEND_RAW, dev_mode=DEV_MODE
)


def validate_checkpoint_settings(
    backend: str, *, explicit_url: str = "", database_url: str = "", dev_mode: bool = False
) -> list[str]:
    """校验 checkpoint 配置，返回错误信息列表（空 = 通过）。

    生产（``dev_mode=False``）：
      - 禁止 ``memory``（进程内、不跨 worker/副本、重启丢失）；
      - ``postgres`` 必须能解析出 PostgreSQL DSN（显式 URL 或 DATABASE_URL）。
    开发/测试：不做约束（允许显式 memory / 自动 memory）。
    """
    problems: list[str] = []
    if dev_mode:
        return problems
    if backend == "memory":
        problems.append(
            "Production requires LANGGRAPH_CHECKPOINT_BACKEND=postgres "
            "(MemorySaver is process-local and not shared across workers/replicas)"
        )
    elif backend == "postgres" and not derive_checkpoint_database_url(
        explicit_url, database_url=database_url
    ):
        problems.append(
            "Production checkpoint backend=postgres requires a PostgreSQL "
            "LANGGRAPH_CHECKPOINT_DATABASE_URL or a PostgreSQL DATABASE_URL"
        )
    return problems


# ===== 分布式 Agent Runtime（异步 Run + Celery Worker）=====
# AgentRun 业务状态真相源是数据库；Redis/Celery 仅调度。
#
# ---- 执行模式（canonical 开关）----
#
# AGENT_EXECUTION_MODE 决定的是**异步 Run 路径**（``POST /api/runs``）在哪里执行：
#   inline : API 进程内执行。适合开发/测试；**不是** durable —— 进程崩溃该 run
#            直接丢失，且没有 worker 重投 / lease 接管 / 事件流。
#   queued : 落库 + 入队，由独立 Celery worker 执行。生产推荐。
#
# 它**不影响** 快路径 ``/api/chat``、``/api/chat/stream``、``/api/chat/multimodal``：
# 那些接口是既有的**兼容性 inline 路径**，始终在 API 进程内执行 LangGraph
# （见 ``api/app.py::_run_graph``）。把流式接口改走队列会改变延迟与响应结构，
# 因此本仓库**不做**这种改写；不要宣称"所有请求都经过 Worker"。
#
# AGENT_RUN_DISPATCH 是同义的历史变量（celery=inline 的旧命名）。未显式设置
# AGENT_EXECUTION_MODE 时由它派生，两者冲突时报错，避免出现两个互相矛盾的旋钮。
AGENT_EXECUTION_MODE_RAW = os.getenv("AGENT_EXECUTION_MODE", "").strip().lower()
AGENT_RUN_DISPATCH_RAW = os.getenv("AGENT_RUN_DISPATCH", "").strip().lower()

# dispatch: celery（生产，解耦到 worker）| inline（开发/测试 fallback，进程内执行）
AGENT_RUN_DISPATCH = AGENT_RUN_DISPATCH_RAW or "celery"
AGENT_RUN_QUEUE = os.getenv("AGENT_RUN_QUEUE", "agent_runs").strip() or "agent_runs"
AGENT_RUN_MAX_ATTEMPTS = _int_env("AGENT_RUN_MAX_ATTEMPTS", 3)
AGENT_RUN_TASK_SOFT_TIME_LIMIT = _int_env("AGENT_RUN_TASK_SOFT_TIME_LIMIT", 120)
AGENT_RUN_TASK_TIME_LIMIT = _int_env("AGENT_RUN_TASK_TIME_LIMIT", 180)
# Redis broker 未 ACK 消息的 visibility timeout（秒）；需 > 最长任务时间
AGENT_RUN_VISIBILITY_TIMEOUT = _int_env("AGENT_RUN_VISIBILITY_TIMEOUT", 3600)
# Celery broker 默认复用 REDIS_URL；result backend 留空表示不落结果（非真相源）
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "").strip() or REDIS_URL
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "").strip()

if AGENT_RUN_DISPATCH not in ("celery", "inline"):
    raise ConfigurationError(
        f"AGENT_RUN_DISPATCH 非法: {AGENT_RUN_DISPATCH!r}（仅支持 celery | inline）"
    )


def _resolve_execution_mode() -> str:
    """归一化 ``AGENT_EXECUTION_MODE`` / ``AGENT_RUN_DISPATCH`` 为单一权威值。

    优先级：显式设置的 ``AGENT_EXECUTION_MODE`` **优先**于历史变量
    ``AGENT_RUN_DISPATCH``。两者不一致时**告警但不拒绝启动**——否则仓库 ``.env``
    里的 ``AGENT_RUN_DISPATCH=inline``（开发默认）会让任何
    ``AGENT_EXECUTION_MODE=queued`` 配置直接起不来。非法取值仍然 fail-fast。
    """
    if not AGENT_EXECUTION_MODE_RAW:
        # 未设置 canonical 旋钮：沿用历史变量
        return "queued" if AGENT_RUN_DISPATCH == "celery" else "inline"
    if AGENT_EXECUTION_MODE_RAW not in ("inline", "queued"):
        raise ConfigurationError(
            f"AGENT_EXECUTION_MODE 非法: {AGENT_EXECUTION_MODE_RAW!r}"
            "（仅支持 inline | queued）"
        )
    expected_dispatch = "celery" if AGENT_EXECUTION_MODE_RAW == "queued" else "inline"
    if AGENT_RUN_DISPATCH_RAW and expected_dispatch != AGENT_RUN_DISPATCH:
        import warnings

        warnings.warn(
            f"AGENT_EXECUTION_MODE={AGENT_EXECUTION_MODE_RAW!r} 覆盖 "
            f"AGENT_RUN_DISPATCH={AGENT_RUN_DISPATCH!r}（canonical 旋钮优先；"
            "AGENT_RUN_DISPATCH 仅作向后兼容，建议只保留 AGENT_EXECUTION_MODE）",
            RuntimeWarning,
            stacklevel=2,
        )
    return AGENT_EXECUTION_MODE_RAW


AGENT_EXECUTION_MODE = _resolve_execution_mode()
# canonical 模式是权威值，据此同步实际派发方式（worker 执行入口读 AGENT_RUN_DISPATCH）
AGENT_RUN_DISPATCH = "celery" if AGENT_EXECUTION_MODE == "queued" else "inline"

#: 快路径接口始终 inline（不由 AGENT_EXECUTION_MODE 控制）。
#: 仅用于文档/健康检查自述，避免出现"所有请求都经过 Worker"的误读。
AGENT_INLINE_COMPAT_ENDPOINTS = ("/api/chat", "/api/chat/stream", "/api/chat/multimodal")
AGENT_QUEUED_RUN_ENDPOINTS = ("/api/runs",)

# Thread 串行执行锁（同一 thread 的 Run 不得并发；不同 thread 可并发）
AGENT_RUN_THREAD_LOCK_ENABLED = os.getenv("AGENT_RUN_THREAD_LOCK_ENABLED", "true").lower() == "true"
AGENT_RUN_THREAD_LOCK_BACKEND = os.getenv("AGENT_RUN_THREAD_LOCK_BACKEND", "redis").strip().lower()
AGENT_RUN_THREAD_LOCK_TTL_SECONDS = _float_env("AGENT_RUN_THREAD_LOCK_TTL_SECONDS", 300.0)
AGENT_RUN_THREAD_LOCK_RETRY_DELAY_SECONDS = _float_env(
    "AGENT_RUN_THREAD_LOCK_RETRY_DELAY_SECONDS", 5.0
)
# API 执行边界获取锁的最大等待时间（秒）；超时返回 THREAD_BUSY（HTTP 409）
AGENT_RUN_THREAD_LOCK_ACQUIRE_TIMEOUT_SECONDS = _float_env(
    "AGENT_RUN_THREAD_LOCK_ACQUIRE_TIMEOUT_SECONDS", 5.0
)
# worker ownership lease / heartbeat（崩溃后 lease 过期可被接管）
AGENT_RUN_LEASE_SECONDS = _float_env("AGENT_RUN_LEASE_SECONDS", 180.0)
AGENT_RUN_HEARTBEAT_SECONDS = _float_env("AGENT_RUN_HEARTBEAT_SECONDS", 30.0)
# 重试退避（指数 + jitter）
AGENT_RUN_RETRY_BASE_DELAY = _float_env("AGENT_RUN_RETRY_BASE_DELAY", 2.0)
AGENT_RUN_RETRY_MAX_DELAY = _float_env("AGENT_RUN_RETRY_MAX_DELAY", 60.0)
AGENT_RUN_RETRY_JITTER = _float_env("AGENT_RUN_RETRY_JITTER", 0.3)

# ---------------------------------------------------------------------------
# Human-in-the-loop（高风险工具副作用的人工审批治理）
# ---------------------------------------------------------------------------
# 总开关。关闭时 HIGH 风险工具**直接执行**（不拦），仅 MEDIUM/LOW 记录。
# 生产建议保持 true：关闭等于主动放弃该治理边界。
HITL_ENABLED = os.getenv("HITL_ENABLED", "false").lower() == "true"

# 强制人工审批的工具名（逗号分隔，大小写不敏感）
HITL_HIGH_RISK_TOOLS = os.getenv("HITL_HIGH_RISK_TOOLS", "")
# 只记录不拦截的工具名（逗号分隔）
HITL_MEDIUM_RISK_TOOLS = os.getenv("HITL_MEDIUM_RISK_TOOLS", "")

# 金额阈值：提案参数中任一金额字段 >= 该值即判为 HIGH（覆盖工具名白名单之外的
# 大额写操作）。0 = 关闭金额维度。
HITL_HIGH_AMOUNT_THRESHOLD = _float_env("HITL_HIGH_AMOUNT_THRESHOLD", 0.0)

# 审批有效期：超时未决策 -> EXPIRED（等同拒绝，但可与「明确拒绝」区分统计）。
HITL_APPROVAL_TTL_SECONDS = _float_env("HITL_APPROVAL_TTL_SECONDS", 3600.0)

# 可审批的角色（4 级 RBAC 中的高级角色）。customer 永不拥有审批权。
HITL_REVIEWER_ROLES = tuple(
    r.strip().lower()
    for r in os.getenv("HITL_REVIEWER_ROLES", "admin,supervisor").split(",")
    if r.strip()
)

# ---------------------------------------------------------------------------
if AGENT_RUN_THREAD_LOCK_BACKEND not in ("redis", "memory"):
    raise ConfigurationError(
        f"AGENT_RUN_THREAD_LOCK_BACKEND 非法: {AGENT_RUN_THREAD_LOCK_BACKEND!r}"
        "（仅支持 redis | memory）"
    )


def validate_distributed_runtime_settings(
    *,
    dev_mode: bool,
    session_backend: str,
    checkpoint_backend: str,
    gunicorn_workers: int,
    lock_enabled: bool,
    lock_backend: str,
    agent_run_dispatch: str = "celery",
    lock_ttl_seconds: float = 300.0,
    task_time_limit_seconds: float = 180.0,
    ttl_safety_margin_seconds: float = 30.0,
) -> list[str]:
    """生产分布式运行时一致性校验（纯函数，返回问题列表，空 = 通过）。

    规则：
      - 生产必须 Redis Session（进程内 memory 在多 worker 下分片、重启丢失）；
      - 生产异步 Run 必须 `AGENT_RUN_DISPATCH=celery`（inline 在 API 进程内执行，
        失去 worker 解耦，不能用于生产）；
      - gunicorn 多 worker 时要求 postgres checkpoint + redis session +
        分布式 thread lock，否则跨 worker 正确性无法保证；
      - thread lock TTL 必须大于 task time limit + safety margin，否则任务仍在执行时
        锁就可能过期（stale worker 边界，见 docs）。
    开发/测试不做约束。
    """
    problems: list[str] = []
    if dev_mode:
        return problems
    if session_backend != "redis":
        problems.append(
            "Production requires SESSION_STORAGE_BACKEND=redis "
            f"(got {session_backend!r}); in-process memory sessions are not shared "
            "across gunicorn workers/replicas and are lost on restart"
        )
    if agent_run_dispatch != "celery":
        problems.append(
            "Production requires AGENT_RUN_DISPATCH=celery "
            f"(got {agent_run_dispatch!r}); inline runs execute inside the API "
            "process and lose worker decoupling"
        )
    if gunicorn_workers > 1:
        if checkpoint_backend != "postgres":
            problems.append(
                f"GUNICORN_WORKERS={gunicorn_workers}>1 requires "
                "LANGGRAPH_CHECKPOINT_BACKEND=postgres (memory checkpoint is per-process)"
            )
        if session_backend != "redis":
            problems.append(
                f"GUNICORN_WORKERS={gunicorn_workers}>1 requires SESSION_STORAGE_BACKEND=redis"
            )
        if not lock_enabled or lock_backend != "redis":
            problems.append(
                f"GUNICORN_WORKERS={gunicorn_workers}>1 requires "
                "AGENT_RUN_THREAD_LOCK_ENABLED=true and "
                "AGENT_RUN_THREAD_LOCK_BACKEND=redis (per-thread cross-process lock)"
            )
    if lock_ttl_seconds <= task_time_limit_seconds + ttl_safety_margin_seconds:
        problems.append(
            "AGENT_RUN_THREAD_LOCK_TTL_SECONDS "
            f"({lock_ttl_seconds}) must exceed AGENT_RUN_TASK_TIME_LIMIT "
            f"({task_time_limit_seconds}) + safety margin ({ttl_safety_margin_seconds}); "
            "otherwise the lock can expire while the task is still running"
        )
    return problems


# ===== v4.1: SSE 流式输出配置 =====
SSE_CHUNK_SIZE = _int_env("SSE_CHUNK_SIZE", 50)  # 每次发送的字符数
SSE_ENABLED = os.getenv("SSE_ENABLED", "true").lower() == "true"

# ===== v4.1: 多模态配置 =====
MULTIMODAL_ENABLED = os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true"
MAX_IMAGE_SIZE_MB = _int_env("MAX_IMAGE_SIZE_MB", 5)
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]

# ===== v5.1: Vision LLM 配置（多模态模型）=====
VISION_MODEL = os.getenv("VISION_MODEL", "")  # 留空则复用 OPENAI_MODEL
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "")  # 留空则复用 OPENAI_BASE_URL
VISION_API_KEY = os.getenv("VISION_API_KEY", "")  # 留空则复用 OPENAI_API_KEY

# ===== v5.1: CLIP 多模态检索配置 =====
CLIP_ENABLED = os.getenv("CLIP_ENABLED", "false").lower() == "true"
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME", "image_knowledge")

# ===== v4.1: Redis 用途扩展 =====
REDIS_JWT_PREFIX = os.getenv("REDIS_JWT_PREFIX", "csai:jwt:blacklist:")
REDIS_RATE_PREFIX = os.getenv("REDIS_RATE_PREFIX", "csai:rate:")
REDIS_SESSION_PREFIX = os.getenv("REDIS_SESSION_PREFIX", "csai:session:")

# ===== v4.1: A/B 测试配置 =====
AB_TEST_ENABLED = os.getenv("AB_TEST_ENABLED", "false").lower() == "true"

# ===== v4.1: 自我评估配置 =====
EVAL_LOW_SCORE_THRESHOLD = _int_env("EVAL_LOW_SCORE_THRESHOLD", 40)  # 低于此分触发告警
EVAL_ALERT_ENABLED = os.getenv("EVAL_ALERT_ENABLED", "true").lower() == "true"

# ===== v4.3: 低分重试与模式升级配置 =====
EVAL_RETRY_THRESHOLD = _int_env("EVAL_RETRY_THRESHOLD", 30)  # 低于此分触发自动重试/升级
EVAL_RETRY_ENABLED = os.getenv("EVAL_RETRY_ENABLED", "true").lower() == "true"
MODE_UPGRADE_ENABLED = os.getenv("MODE_UPGRADE_ENABLED", "true").lower() == "true"

# ===== v4.3: 各协作模式独立 SLA 超时（秒）=====
SLA_SEQUENTIAL_MAX = _float_env("SLA_SEQUENTIAL_MAX", 15.0)
SLA_PARALLEL_MAX = _float_env("SLA_PARALLEL_MAX", 20.0)
SLA_CONSULTATION_MAX = _float_env("SLA_CONSULTATION_MAX", 25.0)
SLA_HIERARCHICAL_MAX = _float_env("SLA_HIERARCHICAL_MAX", 30.0)
SLA_REACT_MAX = _float_env("SLA_REACT_MAX", 30.0)


def validate_mcp_settings(
    *,
    enabled: bool = MCP_ENABLED,
    servers: str = MCP_SERVERS,
    max_payload_bytes: int = MCP_MAX_PAYLOAD_BYTES,
) -> list[str]:
    """校验 MCP 配置（纯函数，无 IO），返回错误信息列表（空 = 通过）。

    只做**结构性**检查，不去连接任何 server：这里的职责是让明显不可用的配置在
    启动时炸掉，而不是等到第一次工具调用才失败。

    刻意不检查的东西：``MCP_SERVERS`` 里的每条记录是否合法由
    ``tools.mcp_adapter.load_mcp_server_configs`` 负责（它对单条非法记录是
    跳过 + 记日志，而非整体拒绝），避免同一个 JSON 在两处用两套规则解析。
    """
    if not enabled:
        return []

    errors: list[str] = []
    text = (servers or "").strip()
    if not text:
        errors.append("MCP_ENABLED=true 但 MCP_SERVERS 为空：没有 allowlist 就没有可连接的 server")
        return errors
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        errors.append(f"MCP_SERVERS 不是合法 JSON: {type(e).__name__}")
        return errors
    if not isinstance(data, list):
        errors.append(f"MCP_SERVERS 必须是 JSON 数组，实际是 {type(data).__name__}")
        return errors
    if not data:
        errors.append("MCP_SERVERS 是空数组：等于没有声明任何 server")
        return errors
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            errors.append("MCP_SERVERS 中存在缺少 name 的条目")
            break
        errors.extend(_validate_mcp_risk_level(item))
    if max_payload_bytes <= 0:
        errors.append(f"MCP_MAX_PAYLOAD_BYTES 必须为正数（当前 {max_payload_bytes}）")
    return errors


#: MCP server 允许显式声明的风险等级。**与 ``core.hitl.risk.RiskLevel`` 同源**，
#: 这里只做字面量校验，不另立词汇表。
MCP_RISK_LEVELS = ("low", "medium", "high")


def _validate_mcp_risk_level(item: dict) -> list[str]:
    """MCP server ``risk_level`` 的结构性校验（fail closed 的方向是"响亮地失败"）。

    运行时（``tools.mcp_adapter._coerce_risk_level``）会把缺失 / 非法的
    ``risk_level`` 一律收敛到 ``HIGH``，绝不会放行。这里额外在**启动期**报出来，
    是为了让"我以为配了 low、实际拼错了"或"压根忘了配"这两种错误在部署时就可见，
    而不是等到工具列表莫名为空才发现。
    """
    name = item.get("name")
    raw = item.get("risk_level")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return [
            f"MCP server {name!r} 未显式声明 risk_level —— 按 'high' 处理，"
            '该 server 的工具不会被注册；若确认只读请加 "risk_level": "low"'
        ]
    if not isinstance(raw, str) or raw.strip().lower() not in MCP_RISK_LEVELS:
        return [
            f"MCP server {name!r} 的 risk_level 非法（{raw!r}）—— 期望 "
            f"{MCP_RISK_LEVELS} 之一；将按 'high' 处理（工具不会被注册）"
        ]
    return []


# ===== P0-3: 生产环境关键配置启动校验 =====
def validate_required_config():
    """生产环境启动时校验关键配置项非空非占位符"""
    warnings = []

    # v4.3 安全加固：生产环境禁止 DEV_MODE
    if _DEV_MODE:
        # 检测是否为生产环境（通过多个信号判断）
        _is_production = (
            os.getenv("APP_MODE", "").lower() == "prod"
            or os.getenv("LOG_FORMAT", "").lower() == "json"
            or os.getenv("ENVIRONMENT", "").lower() == "production"
        )
        if _is_production:
            raise ConfigurationError(
                "🚨 安全错误: 生产环境(APP_MODE=prod/LOG_FORMAT=json/ENVIRONMENT=production)下禁止 DEV_MODE=true！"
                "请设置 DEV_MODE=false 后重启。"
            )
        return  # 非生产环境的开发模式跳过后续校验

    _PLACEHOLDER_PREFIXES = (
        "your-",
        "change-me",
        "change_me",
        "sk-placeholder",
        "sk-xxx",
        "sk-your",
        "sk-test-placeholder",
        "sk-tnwwg",  # 匹配已知旧占位符
    )
    errors = []

    # LLM API Key
    if not OPENAI_API_KEY or any(
        OPENAI_API_KEY.lower().startswith(p) for p in _PLACEHOLDER_PREFIXES
    ):
        errors.append("OPENAI_API_KEY 未配置或使用占位符")

    # ---- 以下均为 **API 进程专属**：worker 不终结 HTTP 请求、不签发/校验用户凭据，
    # 因此不应该因为缺少这些配置而无法启动（见 SERVICE_ROLE / IS_WORKER_PROCESS）。
    if not IS_WORKER_PROCESS:
        # JWT Secret（v5.0: 最小 32 字符，防止弱密钥）
        if not JWT_SECRET or any(
            p in JWT_SECRET.lower() for p in ("change-me", "change_me", "your-", "dev-")
        ):
            errors.append("JWT_SECRET 未配置或使用默认值/弱密钥")
        elif len(JWT_SECRET) < 32:
            errors.append(
                f"JWT_SECRET 长度不足（{len(JWT_SECRET)} < 32），请使用至少 32 字符的随机密钥"
            )

        # Session Token Secret
        if not SESSION_TOKEN_SECRET or any(
            p in SESSION_TOKEN_SECRET.lower()
            for p in ("change-me", "change_me", "your-", "dev-")
        ):
            errors.append("SESSION_TOKEN_SECRET 未配置或使用默认值/弱密钥")
        elif len(SESSION_TOKEN_SECRET) < 32:
            errors.append(
                f"SESSION_TOKEN_SECRET 长度不足（{len(SESSION_TOKEN_SECRET)} < 32），"
                "请使用至少 32 字符的随机密钥"
            )

        if not _DEV_MODE and "*" in CORS_ORIGINS:
            errors.append("Production CORS_ORIGINS must not contain wildcard *")
        if not _DEV_MODE and not CORS_ORIGINS:
            errors.append(
                "Production CORS_ORIGINS is empty — frontend cross-origin requests will "
                "fail. Set CORS_ORIGINS or ALLOWED_ORIGINS"
            )

    if not _DEV_MODE and not DATABASE_URL:
        errors.append("Production requires DATABASE_URL (PostgreSQL)")

    # LangGraph checkpoint：生产禁止进程内 MemorySaver（多 worker/多副本不共享、
    # 重启即丢失），且必须能解析出可用的 PostgreSQL DSN。
    if not _DEV_MODE:
        errors.extend(
            validate_checkpoint_settings(
                LANGGRAPH_CHECKPOINT_BACKEND,
                explicit_url=LANGGRAPH_CHECKPOINT_DATABASE_URL,
                database_url=DATABASE_URL,
                dev_mode=_DEV_MODE,
            )
        )

    # MCP：生产开启时必须给出可用的 server allowlist。MCP 默认关闭，所以这里只在
    # 显式开启时校验；fail closed —— 配错了就拒绝启动，而不是让服务带着「以为 MCP
    # 已生效、实际一个工具都没注册」的状态跑起来。
    errors.extend(
        validate_mcp_settings(
            enabled=MCP_ENABLED,
            servers=MCP_SERVERS,
            max_payload_bytes=MCP_MAX_PAYLOAD_BYTES,
        )
    )

    # P0-2 / P0-6: 生产 session + dispatch + gunicorn 多 worker + lock TTL（纯函数）
    errors.extend(
        validate_distributed_runtime_settings(
            dev_mode=_DEV_MODE,
            session_backend=SESSION_STORAGE_BACKEND,
            checkpoint_backend=LANGGRAPH_CHECKPOINT_BACKEND,
            gunicorn_workers=_int_env("GUNICORN_WORKERS", 0),
            lock_enabled=AGENT_RUN_THREAD_LOCK_ENABLED,
            lock_backend=AGENT_RUN_THREAD_LOCK_BACKEND,
            agent_run_dispatch=AGENT_RUN_DISPATCH,
            lock_ttl_seconds=AGENT_RUN_THREAD_LOCK_TTL_SECONDS,
            task_time_limit_seconds=float(AGENT_RUN_TASK_TIME_LIMIT),
        )
    )

    if errors:
        for err in errors:
            print(f"🚨 配置校验失败: {err}", file=sys.stderr)
        raise ConfigurationError(
            f"生产环境启动失败：{len(errors)} 项关键配置缺失，请检查 .env 文件"
        )

    # Non-fatal warnings for missing optional-but-recommended config
    if not _DEV_MODE and not RAG_PERSIST_DIRECTORY:
        warnings.append("RAG_PERSIST_DIRECTORY not set, vector DB will run in-memory")
    if not _DEV_MODE and not ALERT_WEBHOOKS and not SMTP_HOST:
        warnings.append("No alert notification channels configured")
    if not _DEV_MODE and ERP_MODE == "mock":
        warnings.append("ERP_MODE=mock: using fake ERP data, set to 'real' for production")
    if not _DEV_MODE and not SESSION_ENCRYPTION_KEY:
        warnings.append("SESSION_ENCRYPTION_KEY not set: session data stored in plaintext")
    # v6.3: 当 EMBEDDING_API_KEY 为空（即从 OPENAI_API_KEY 回退）时发出警告
    _embedding_raw = os.getenv("EMBEDDING_API_KEY", "")
    if not _DEV_MODE and not _embedding_raw:
        warnings.append(
            "EMBEDDING_API_KEY not set, reusing OPENAI_API_KEY for embedding service — configure a dedicated key for production"
        )
    if not _DEV_MODE and AGENT_RUN_DISPATCH == "inline":
        warnings.append(
            "AGENT_RUN_DISPATCH=inline in production: runs execute inside the API "
            "process (no worker decoupling). Use 'celery' for production."
        )
    if not _DEV_MODE and AGENT_RUN_THREAD_LOCK_BACKEND == "memory":
        warnings.append(
            "AGENT_RUN_THREAD_LOCK_BACKEND=memory in production: thread lock is "
            "process-local, not shared across workers/replicas. Use 'redis'."
        )

    for w in warnings:
        logging.getLogger("config").warning(f"[config] {w}")


validate_required_config()


# ===== v5.0: FeatureFlags 集中管理 =====
class FeatureFlags:
    """功能开关集中管理，所有开关统一从此处读取"""

    # 多模态
    MULTIMODAL_ENABLED = MULTIMODAL_ENABLED
    # A/B 测试
    AB_TEST_ENABLED = AB_TEST_ENABLED
    # SSE 流式输出
    SSE_ENABLED = SSE_ENABLED
    # 自动重试/模式升级
    EVAL_RETRY_ENABLED = EVAL_RETRY_ENABLED
    MODE_UPGRADE_ENABLED = MODE_UPGRADE_ENABLED
    # Function Calling
    TOOL_MAX_ROUNDS = TOOL_MAX_ROUNDS
    TOOL_RESULT_OPTIMIZATION_ENABLED = TOOL_RESULT_OPTIMIZATION_ENABLED
    TOOL_RESULT_MAX_TOKENS = TOOL_RESULT_MAX_TOKENS
    TOOL_RESULT_MAX_ITEMS = TOOL_RESULT_MAX_ITEMS
    TOOL_RESULT_PRESERVE_RECENT = TOOL_RESULT_PRESERVE_RECENT
    TOOL_RESULT_OFFLOAD_ENABLED = TOOL_RESULT_OFFLOAD_ENABLED
    TOOL_RESULT_OFFLOAD_MIN_TOKENS = TOOL_RESULT_OFFLOAD_MIN_TOKENS
    TOOL_RESULT_STORE_TTL_SECONDS = TOOL_RESULT_STORE_TTL_SECONDS
    TOOL_RESULT_SEMANTIC_SUMMARY_ENABLED = TOOL_RESULT_SEMANTIC_SUMMARY_ENABLED
    TOOL_RESULT_CACHE_ENABLED = TOOL_RESULT_CACHE_ENABLED
    # ReAct 推理
    REACT_COMPLEXITY_THRESHOLD = REACT_COMPLEXITY_THRESHOLD

    @classmethod
    def get_all(cls) -> dict[str, bool | int]:
        """返回所有功能开关的当前状态（用于健康检查/调试）"""
        return {
            "multimodal": cls.MULTIMODAL_ENABLED,
            "ab_test": cls.AB_TEST_ENABLED,
            "sse": cls.SSE_ENABLED,
            "eval_retry": cls.EVAL_RETRY_ENABLED,
            "mode_upgrade": cls.MODE_UPGRADE_ENABLED,
            "tool_max_rounds": cls.TOOL_MAX_ROUNDS,
            "tool_result_optimization": cls.TOOL_RESULT_OPTIMIZATION_ENABLED,
            "tool_result_max_tokens": cls.TOOL_RESULT_MAX_TOKENS,
            "tool_result_max_items": cls.TOOL_RESULT_MAX_ITEMS,
            "tool_result_preserve_recent": cls.TOOL_RESULT_PRESERVE_RECENT,
            "tool_result_offload": cls.TOOL_RESULT_OFFLOAD_ENABLED,
            "tool_result_offload_min_tokens": cls.TOOL_RESULT_OFFLOAD_MIN_TOKENS,
            "tool_result_store_ttl_seconds": cls.TOOL_RESULT_STORE_TTL_SECONDS,
            "tool_result_semantic_summary": cls.TOOL_RESULT_SEMANTIC_SUMMARY_ENABLED,
            "tool_result_cache": cls.TOOL_RESULT_CACHE_ENABLED,
            "react_complexity_threshold": cls.REACT_COMPLEXITY_THRESHOLD,
        }
