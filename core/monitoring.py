"""
监控与基础设施模块（从 multi_agent_customer_service.py 提取）
- MetricsCollector: 性能指标实时采集（KPI + SLA + 解决率）
- CircuitBreaker: LLM 调用熔断器（三态：CLOSED / OPEN / HALF_OPEN）
- SLAAlertManager: SLA 违约率滑动窗口告警

LLM 客户端已迁移到 llm/client.py

v3.4 优化：
- MetricsCollector: asyncio.Lock 保护并发写入
- CircuitBreaker: 状态转换原子化，防止多协程同时探测

v5.4 新增：
- 业务指标监控（用户满意度、Agent使用分布、意图分布）
"""

import asyncio
import contextlib
import json
import time
from collections import deque
from typing import Any

from core.config import (
    CIRCUIT_BREAKER_FAIL_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIME,
    RESPONSE_TIME_TARGET_MAX,
    RESPONSE_TIME_TARGET_MIN,
    SLA_ALERT_COOLDOWN,
    SLA_ALERT_THRESHOLD,
    SLA_ALERT_WINDOW,
)
from core.logger import get_logger

logger = get_logger("monitoring")

# v5.4: 业务指标 Prometheus 监控
try:
    from prometheus_client import REGISTRY
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Gauge as _PromGauge
    from prometheus_client import Histogram as _PromHistogram

    _METRICS: dict[str, object] = {}

    def _counter(name, documentation, *args, **kwargs):
        """Create or retrieve a Counter metric (safe for re-import/test sessions)."""
        if name not in _METRICS:
            try:
                _METRICS[name] = _PromCounter(name, documentation, *args, **kwargs)
            except ValueError:
                with contextlib.suppress(KeyError, ValueError):
                    REGISTRY.unregister(name)
                _METRICS[name] = _PromCounter(name, documentation, *args, **kwargs)
        return _METRICS[name]

    def _gauge(name, documentation, *args, **kwargs):
        """Create or retrieve a Gauge metric (safe for re-import/test sessions)."""
        if name not in _METRICS:
            try:
                _METRICS[name] = _PromGauge(name, documentation, *args, **kwargs)
            except ValueError:
                with contextlib.suppress(KeyError, ValueError):
                    REGISTRY.unregister(name)
                _METRICS[name] = _PromGauge(name, documentation, *args, **kwargs)
        return _METRICS[name]

    def _histogram(name, documentation, *args, **kwargs):
        """Create or retrieve a Histogram metric (safe for re-import/test sessions)."""
        if name not in _METRICS:
            try:
                _METRICS[name] = _PromHistogram(name, documentation, *args, **kwargs)
            except ValueError:
                with contextlib.suppress(KeyError, ValueError):
                    REGISTRY.unregister(name)
                _METRICS[name] = _PromHistogram(name, documentation, *args, **kwargs)
        return _METRICS[name]

    # 用户满意度评分分布
    user_satisfaction_score = _histogram(
        'user_satisfaction_score',
        'User satisfaction score distribution (1-5)',
        buckets=[1, 2, 3, 4, 5],
    )

    # Agent 使用次数统计
    agent_usage_total = _counter(
        'agent_usage_total',
        'Agent usage count by type',
        ['agent_type'],
    )

    # 查询意图分布
    intent_distribution_total = _counter(
        'intent_distribution_total',
        'Query intent distribution',
        ['intent_type'],
    )

    # 协作模式使用统计
    collaboration_mode_total = _counter(
        'collaboration_mode_total',
        'Collaboration mode usage count',
        ['mode_name'],
    )

    # 会话解决率
    session_resolution_rate = _gauge(
        'session_resolution_rate',
        'Session resolution rate (resolved / total)',
    )

    # 人工升级率
    escalation_rate = _gauge(
        'escalation_rate',
        'Human escalation rate (escalated / total)',
    )

    # 缓存命中率（业务维度）
    business_cache_hit_rate = _gauge(
        'business_cache_hit_rate',
        'Business-level cache hit rate',
    )

    # v6.1: 缓存分层命中率
    cache_l1_hits_total = _counter('cache_l1_hits_total', 'L1 exact-match cache hits')
    cache_l2_hits_total = _counter('cache_l2_hits_total', 'L2 semantic cache hits')
    cache_l3_hits_total = _counter('cache_l3_hits_total', 'L3 jaccard fallback cache hits')
    cache_fallback_total = _counter('cache_fallback_total', 'Fallback to L3 Jaccard')

    # v6.1: 流式响应性能
    stream_ttfb_seconds = _histogram(
        'stream_ttfb_seconds',
        'Time to first byte in streaming responses',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    )

    # v6.1: Trace 跟踪
    trace_spans_total = _counter('trace_spans_total', 'Total trace spans')

    # v6.1: RAG 检索
    rag_queries_total = _counter('rag_queries_total', 'RAG queries count')
    rag_recall_at_3 = _gauge('rag_recall_at_3', 'RAG recall@3 score')

    # v6.1: 场景路由
    scene_routing_total = _counter(
        'scene_routing_total',
        'Scene routing count',
        ['scene_name'],
    )

    # v6.1: DI 组件
    active_components_total = _gauge('active_components_total', 'Active DI components')

    # v6.1 收尾: RAG 搜索延迟
    rag_search_latency_seconds = _histogram(
        "rag_search_latency_seconds",
        "RAG search query latency distribution",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )

    # v6.1 收尾: Agent 处理时间
    agent_process_time_seconds = _histogram(
        "agent_process_time_seconds",
        "Agent processing time per call",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )

    # v6.1 收尾: 缓存写入统计
    cache_writes_total = _counter(
        "cache_writes_total",
        "Total number of cache write operations",
    )
    embedding_provider_failures_total = _counter(
        "embedding_provider_failures_total",
        "Embedding provider unavailable events",
        ["reason"],
    )
    embedding_dimension_errors_total = _counter(
        "embedding_dimension_errors_total",
        "Embedding vectors rejected for invalid dimension or values",
    )
    vector_channel_disabled_total = _counter(
        "vector_channel_disabled_total",
        "Retrieval runs where the vector channel was disabled",
    )
    bm25_fallback_used_total = _counter(
        "bm25_fallback_used_total",
        "Retrieval runs using BM25-only fallback",
    )
    retrieval_no_channel_total = _counter(
        "retrieval_no_channel_total",
        "Retrieval runs with no usable channel",
    )
    bm25_rebuilds_total = _counter("bm25_rebuilds_total", "Successful BM25 rebuilds")
    bm25_rebuild_failures_total = _counter(
        "bm25_rebuild_failures_total", "Failed BM25 rebuilds", ["reason"]
    )
    semantic_cache_embedding_failures_total = _counter(
        "semantic_cache_embedding_failures_total",
        "Semantic cache (L2) tiers skipped due to embedding failure",
    )
    tool_result_raw_bytes = _histogram(
        "tool_result_raw_bytes", "Raw serialized tool result size in bytes", buckets=[100, 500, 1000, 5000, 10000, 50000]
    )
    tool_result_optimized_bytes = _histogram(
        "tool_result_optimized_bytes", "Optimized serialized tool result size in bytes", buckets=[100, 500, 1000, 5000, 10000, 50000]
    )
    tool_result_tokens_before = _histogram(
        "tool_result_tokens_before", "Estimated tool result tokens before optimization", buckets=[10, 50, 100, 500, 1000, 5000]
    )
    tool_result_tokens_after = _histogram(
        "tool_result_tokens_after", "Estimated tool result tokens after optimization", buckets=[10, 50, 100, 500, 1000, 5000]
    )
    tool_result_optimized_total = _counter(
        "tool_result_optimized_total", "Tool results processed by the context optimizer", ["tool_name"]
    )
    tool_result_truncated_total = _counter(
        "tool_result_truncated_total", "Tool results truncated by the context optimizer", ["tool_name"]
    )
    tool_result_offloaded_total = _counter("tool_result_offloaded_total", "Tool results offloaded to external storage")
    tool_result_recovered_total = _counter("tool_result_recovered_total", "Tool results recovered from external storage")
    tool_result_recovery_failed_total = _counter("tool_result_recovery_failed_total", "Tool result recovery failures")
    tool_result_store_latency_seconds = _histogram("tool_result_store_latency_seconds", "Tool result store latency")
    tool_result_store_errors_total = _counter("tool_result_store_errors_total", "Tool result store errors")
    tool_result_summary_total = _counter("tool_result_summary_total", "Tool result summaries attempted")
    tool_result_summary_failed_total = _counter("tool_result_summary_failed_total", "Tool result summary failures")
    tool_result_compressor_total = _counter("tool_result_compressor_total", "Tool result compressor strategies", ["strategy"])
    tool_result_cache_requests_total = _counter("tool_result_cache_requests_total", "Tool result cache requests", ["tool_name", "outcome"])
    tool_result_cache_hits_total = _counter("tool_result_cache_hits_total", "Tool result cache hits", ["tool_name"])
    tool_result_cache_misses_total = _counter("tool_result_cache_misses_total", "Tool result cache misses", ["tool_name"])
    tool_result_cache_writes_total = _counter("tool_result_cache_writes_total", "Tool result cache writes", ["tool_name"])
    tool_result_cache_errors_total = _counter("tool_result_cache_errors_total", "Tool result cache errors", ["tool_name"])
    tool_result_cache_bypass_total = _counter("tool_result_cache_bypass_total", "Tool result cache bypasses", ["tool_name"])
    tool_result_cache_latency_seconds = _histogram("tool_result_cache_latency_seconds", "Tool result cache lookup latency")

    # 分布式 Agent Run 可靠性 + 可观测性（未发布版本；runtime 版本仍为 6.3）
    #
    # label 基数纪律：只允许低基数维度（status / mode / agent / error_type）。
    # 严禁把 run_id / thread_id / user_id / query 放进 label —— 这些是每请求唯一值，
    # 会让 Prometheus 时序数无界增长。
    agent_run_total = _counter("agent_run_total", "AgentRun outcomes by status", ["status"])
    agent_run_retry_total = _counter("agent_run_retry_total", "AgentRun retries scheduled")
    agent_run_dead_letter_total = _counter(
        "agent_run_dead_letter_total", "AgentRun retry exhausted -> DEAD_LETTER"
    )
    agent_run_retry_publication_failure_total = _counter(
        "agent_run_retry_publication_failure_total",
        "Delayed-retry publication failures (task escaped to force broker redelivery)",
    )
    agent_run_dead_letter_replay_total = _counter(
        "agent_run_dead_letter_replay_total",
        "Operator replays of DEAD_LETTER runs (redrive)",
    )
    agent_run_duration_seconds = _histogram(
        "agent_run_duration_seconds",
        "AgentRun execution duration (RUNNING -> terminal)",
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    )
    agent_run_queue_wait_seconds = _histogram(
        "agent_run_queue_wait_seconds",
        "AgentRun queue wait (queued_at -> started_at)",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 300],
    )
    agent_worker_active = _gauge("agent_worker_active", "Active worker tasks")
    agent_worker_task_total = _counter(
        "agent_worker_task_total", "Worker task outcomes", ["status"]
    )
    agent_run_inflight = _gauge("agent_run_inflight", "AgentRuns currently executing")
    agent_run_failed_total = _counter(
        "agent_run_failed_total", "AgentRun terminal failures (FAILED/DEAD_LETTER)"
    )
    agent_thread_lease_acquire_total = _counter(
        "agent_thread_lease_acquire_total", "Thread lease acquisitions"
    )
    agent_thread_lease_contention_total = _counter(
        "agent_thread_lease_contention_total", "Same-thread lease contention deferrals"
    )
    agent_thread_lease_wait_seconds = _histogram(
        "agent_thread_lease_wait_seconds",
        "Time spent acquiring the thread lease",
        buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    )
    agent_thread_lease_renewed_total = _counter(
        "agent_thread_lease_renewed_total",
        "Thread lease TTL renewals during execution",
        ["outcome"],
    )
    agent_worker_heartbeat = _counter(
        "agent_worker_heartbeat", "Worker ownership lease heartbeats", ["outcome"]
    )
    checkpoint_errors_total = _counter(
        "checkpoint_errors_total", "LangGraph checkpoint backend errors"
    )
    agent_checkpoint_recovery_total = _counter(
        "agent_checkpoint_recovery_total",
        "LangGraph executions resumed from an existing checkpoint",
        ["mode"],
    )
    agent_run_idempotency_hit_total = _counter(
        "agent_run_idempotency_hit_total", "HTTP idempotency key hits on run creation"
    )
    agent_tool_idempotency_hit_total = _counter(
        "agent_tool_idempotency_hit_total", "Side-effect tool idempotency hits"
    )
    # human-in-the-loop 审批治理。
    #
    # label 基数纪律同上：只用 risk_level / decision / outcome 这类低基数维度。
    # ``action``（工具名）是**有界**集合（注册表里的工具），但为避免第三方插件
    # 动态注册导致基数无界，action 只进 log 与审批表，不进 Prometheus label。
    agent_approval_requested_total = _counter(
        "agent_approval_requested_total",
        "High-risk side effects gated for human approval",
        ["risk_level"],
    )
    agent_approval_decided_total = _counter(
        "agent_approval_decided_total",
        "Human approval decisions recorded",
        ["decision"],
    )
    # MCP 外部工具接入（默认关闭；MCP_ENABLED=true 才产生时序）。
    #
    # label 基数纪律：``tool`` 只取 **allowlist 内**的工具名，``server`` 只取
    # allowlist 里显式声明的 server 名。两者都是**配置决定**的有界集合，所以可作
    # label。被拒绝的远端工具名（不在 allowlist）一律打成 ``tool="*"``，绝不把
    # 远端提供的字符串变成 label —— 否则 server 一暴露新工具就能无界扩张时序数。
    mcp_tool_call_total = _counter(
        "mcp_tool_call_total", "MCP tool invocations by outcome", ["server", "tool", "status"]
    )
    mcp_tool_error_total = _counter(
        "mcp_tool_error_total", "MCP tool errors by reason", ["server", "tool", "reason"]
    )
    # label 名必须与 tools/mcp_adapter.py::_observe 的调用处**逐字一致**。
    # 声明与观测不一致时，真实 prometheus_client 的 ``labels()`` 会抛
    # ``ValueError: No label names were set when constructing histogram:...``；
    # 该异常被指标降级路径 ``except Exception: pass`` 吞掉，于是直方图**静默零样本**
    # （计数器也一样）。声明侧少了 label 不会有任何报错，只会丢数据。
    # 契约由 tests/unit/test_mcp_metrics_contract.py 锁定（真实 client 断言样本产生）。
    mcp_tool_duration_seconds = _histogram(
        "mcp_tool_duration_seconds",
        "MCP tool invocation latency",
        ["server", "tool"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
    )
    mcp_tool_register_total = _counter(
        "mcp_tool_register_total",
        "MCP tool registration outcomes (registered / skipped_not_low_risk / skipped_collision)",
        ["server", "outcome"],
    )
    agent_approval_pending = _gauge(
        "agent_approval_pending", "Applications awaiting human decision"
    )
    agent_approval_wait_seconds = _histogram(
        "agent_approval_wait_seconds",
        "Time from approval request to human decision",
        buckets=[5, 30, 60, 300, 900, 1800, 3600, 14400],
    )
    agent_approval_expired_total = _counter(
        "agent_approval_expired_total",
        "Approvals auto-expired past their TTL without a decision",
    )
    agent_approval_execution_total = _counter(
        "agent_approval_execution_total",
        "Outcomes of approved side-effect executions",
        ["outcome"],
    )

    PROMETHEUS_BUSINESS_ENABLED = True
except ImportError:
    # Prometheus 未安装，降级为无操作
    class _NoopMetric:
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    user_satisfaction_score = _NoopMetric()
    agent_usage_total = _NoopMetric()
    intent_distribution_total = _NoopMetric()
    collaboration_mode_total = _NoopMetric()
    session_resolution_rate = _NoopMetric()
    escalation_rate = _NoopMetric()
    business_cache_hit_rate = _NoopMetric()
    cache_l1_hits_total = _NoopMetric()
    cache_l2_hits_total = _NoopMetric()
    cache_l3_hits_total = _NoopMetric()
    cache_fallback_total = _NoopMetric()
    stream_ttfb_seconds = _NoopMetric()
    trace_spans_total = _NoopMetric()
    rag_queries_total = _NoopMetric()
    rag_recall_at_3 = _NoopMetric()
    scene_routing_total = _NoopMetric()
    active_components_total = _NoopMetric()
    rag_search_latency_seconds = _NoopMetric()
    agent_process_time_seconds = _NoopMetric()
    cache_writes_total = _NoopMetric()
    embedding_provider_failures_total = _NoopMetric()
    embedding_dimension_errors_total = _NoopMetric()
    vector_channel_disabled_total = _NoopMetric()
    bm25_fallback_used_total = _NoopMetric()
    retrieval_no_channel_total = _NoopMetric()
    bm25_rebuilds_total = _NoopMetric()
    bm25_rebuild_failures_total = _NoopMetric()
    semantic_cache_embedding_failures_total = _NoopMetric()
    tool_result_raw_bytes = _NoopMetric()
    tool_result_optimized_bytes = _NoopMetric()
    tool_result_tokens_before = _NoopMetric()
    tool_result_tokens_after = _NoopMetric()
    tool_result_optimized_total = _NoopMetric()
    tool_result_truncated_total = _NoopMetric()
    tool_result_offloaded_total = _NoopMetric()
    tool_result_recovered_total = _NoopMetric()
    tool_result_recovery_failed_total = _NoopMetric()
    tool_result_store_latency_seconds = _NoopMetric()
    tool_result_store_errors_total = _NoopMetric()
    tool_result_summary_total = _NoopMetric()
    tool_result_summary_failed_total = _NoopMetric()
    tool_result_compressor_total = _NoopMetric()
    tool_result_cache_requests_total = _NoopMetric()
    tool_result_cache_hits_total = _NoopMetric()
    tool_result_cache_misses_total = _NoopMetric()
    tool_result_cache_writes_total = _NoopMetric()
    tool_result_cache_errors_total = _NoopMetric()
    tool_result_cache_bypass_total = _NoopMetric()
    tool_result_cache_latency_seconds = _NoopMetric()
    agent_run_total = _NoopMetric()
    agent_run_retry_total = _NoopMetric()
    agent_run_dead_letter_total = _NoopMetric()
    agent_run_retry_publication_failure_total = _NoopMetric()
    agent_run_dead_letter_replay_total = _NoopMetric()
    agent_run_duration_seconds = _NoopMetric()
    agent_run_queue_wait_seconds = _NoopMetric()
    agent_worker_active = _NoopMetric()
    agent_worker_task_total = _NoopMetric()
    agent_run_inflight = _NoopMetric()
    agent_run_failed_total = _NoopMetric()
    agent_thread_lease_acquire_total = _NoopMetric()
    agent_thread_lease_contention_total = _NoopMetric()
    agent_thread_lease_wait_seconds = _NoopMetric()
    agent_thread_lease_renewed_total = _NoopMetric()
    agent_worker_heartbeat = _NoopMetric()
    checkpoint_errors_total = _NoopMetric()
    agent_checkpoint_recovery_total = _NoopMetric()
    agent_run_idempotency_hit_total = _NoopMetric()
    agent_tool_idempotency_hit_total = _NoopMetric()
    agent_approval_requested_total = _NoopMetric()
    agent_approval_decided_total = _NoopMetric()
    agent_approval_pending = _NoopMetric()
    agent_approval_wait_seconds = _NoopMetric()
    agent_approval_expired_total = _NoopMetric()
    agent_approval_execution_total = _NoopMetric()
    mcp_tool_call_total = _NoopMetric()
    mcp_tool_error_total = _NoopMetric()
    mcp_tool_duration_seconds = _NoopMetric()
    mcp_tool_register_total = _NoopMetric()
    PROMETHEUS_BUSINESS_ENABLED = False


def record_tool_result_optimization(
    tool_name: str, raw_size: int, optimized_size: int, raw_tokens: int, optimized_tokens: int, truncated: bool
) -> None:
    """Record bounded, content-free Tool Result optimization metrics."""
    tool_result_raw_bytes.observe(raw_size)
    tool_result_optimized_bytes.observe(optimized_size)
    tool_result_tokens_before.observe(raw_tokens)
    tool_result_tokens_after.observe(optimized_tokens)
    tool_result_optimized_total.labels(tool_name=tool_name).inc()
    if truncated:
        tool_result_truncated_total.labels(tool_name=tool_name).inc()


def record_tool_result_event(event: str, *, latency_seconds: float | None = None, strategy: str | None = None) -> None:
    """Record content-free V2 events; failures in metrics never escape."""
    try:
        metric = {
            "offloaded": tool_result_offloaded_total,
            "recovered": tool_result_recovered_total,
            "recovery_failed": tool_result_recovery_failed_total,
            "store_error": tool_result_store_errors_total,
            "summary": tool_result_summary_total,
            "summary_failed": tool_result_summary_failed_total,
        }.get(event)
        if metric:
            metric.inc()
        if latency_seconds is not None:
            tool_result_store_latency_seconds.observe(latency_seconds)
        if strategy:
            tool_result_compressor_total.labels(strategy=strategy).inc()
    except Exception:
        return


def record_tool_result_cache_event(tool_name: str, outcome: str, *, latency_seconds: float | None = None) -> None:
    """Record bounded cache outcomes; no arguments, identities, or keys are labels."""
    try:
        tool_result_cache_requests_total.labels(tool_name=tool_name, outcome=outcome).inc()
        if outcome == "hit":
            tool_result_cache_hits_total.labels(tool_name=tool_name).inc()
        elif outcome == "miss":
            tool_result_cache_misses_total.labels(tool_name=tool_name).inc()
        elif outcome == "write":
            tool_result_cache_writes_total.labels(tool_name=tool_name).inc()
        elif outcome == "error":
            tool_result_cache_errors_total.labels(tool_name=tool_name).inc()
        elif outcome == "bypass":
            tool_result_cache_bypass_total.labels(tool_name=tool_name).inc()
        if latency_seconds is not None:
            tool_result_cache_latency_seconds.observe(latency_seconds)
    except Exception:
        return

# ===== 性能指标常量 =====
RESPONSE_TIMES_MAXLEN = 200  # 响应时间 deque 最大长度
STATS_RECENT_COUNT = 100  # 统计时取最近 N 次响应时间
P95_MIN_SAMPLES = 20  # 计算 P95 最少样本数
SESSION_TTL = 3600.0  # 会话统计过期时间（秒，1 小时）
CLEANUP_INTERVAL = 100  # 每 N 次请求清理一次过期会话
METRICS_SNAPSHOT_TTL = 86400  # Redis 快照 TTL（秒，24 小时）
METRICS_HISTORY_MAX = 24  # Redis 历史快照保留数
PERCENTAGE_MULTIPLIER = 100  # 百分比乘数
P95_PERCENTILE = 0.95  # P95 百分位


# ===== 性能指标采集器 =====
# Responsibilities are cohesive: collection, computation, and optional persistence
# all operate on the same in-memory counters. Not splitting.
class MetricsCollector:
    """系统性能指标实时采集（v3.4: asyncio.Lock 保护并发安全）"""

    # --- Section: State & Initialization ---
    def __init__(self):
        self._lock = asyncio.Lock()  # P1-2: 直接初始化，消除懒初始化竞态
        self.total_requests = 0
        self.total_errors = 0
        self.response_times = deque(maxlen=RESPONSE_TIMES_MAXLEN)  # 自动截断，保留最近 N 条
        self.agent_call_counts: dict[str, int] = {}
        self.mode_counts: dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        # SLA 追踪
        self.sla_violations: int = 0
        self.sla_too_fast: int = 0
        # Business KPI tracking（v3.2: 增强版）
        self.total_single_turn_resolved = 0
        self.total_ai_handled = 0
        self.total_escalated = 0
        self.session_turn_counts: dict[str, int] = {}
        self.session_last_activity: dict[str, float] = {}
        self._session_ttl: float = SESSION_TTL  # 会话统计过期时间
        # v3.2: 细粒度解决率追踪
        self.resolution_counts: dict[str, int] = {
            "resolved": 0,
            "uncertain": 0,
            "failed": 0,
            "escalated": 0,
        }
        # v3.2: SLA 告警滑动窗口
        self._sla_window = deque(maxlen=SLA_ALERT_WINDOW)  # 自动截断
        # v5.0: 查询日志（用于热门问题和质量趋势）
        self._query_log: deque = deque(maxlen=1000)  # (timestamp, query, query_type, score, agent)
        # v5.0: 反馈分类统计
        self._feedback_by_category: dict[
            str, dict[str, int]
        ] = {}  # category -> {positive, negative}

    def _ensure_lock(self):
        """P1-2: Lock 已在 __init__ 中初始化，直接返回"""
        return self._lock

    # --- Section: Metric Recording (write path) ---
    async def record_request(
        self,
        elapsed: float,
        agent: str = "",
        mode: str = "",
        cached: bool = False,
        error: bool = False,
        session_id: str = None,
        escalated: bool = False,
        resolution_status: str = "",
        query_type: str = "",  # v5.4: 查询类型（product/billing/complaint等）
        satisfaction_score: int = 0,  # v5.4: 用户满意度评分（1-5）
    ):
        """
        v3.4: 改为 async，使用 asyncio.Lock 保护并发写入
        v5.4: 新增业务指标记录（query_type, satisfaction_score）

        Args:
            elapsed: 请求耗时（秒）
            agent: 处理的Agent名称
            mode: 协作模式名称
            cached: 是否命中缓存
            error: 是否发生错误
            session_id: 会话ID
            escalated: 是否人工升级
            resolution_status: 解决状态 (resolved/uncertain/failed/escalated)
            query_type: 查询类型 (product_info/billing/tech_support等)
            satisfaction_score: 用户满意度评分（1-5）
        """
        async with self._ensure_lock():
            self.total_requests += 1
            if error:
                self.total_errors += 1
            if cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.response_times.append(elapsed)
            # SLA 追踪
            if elapsed > RESPONSE_TIME_TARGET_MAX:
                self.sla_violations += 1
            if elapsed < RESPONSE_TIME_TARGET_MIN:
                self.sla_too_fast += 1
            self._sla_window.append(elapsed)

            # v5.4: 记录Agent使用统计
            if agent:
                self.agent_call_counts[agent] = self.agent_call_counts.get(agent, 0) + 1
                agent_usage_total.labels(agent_type=agent).inc()  # Prometheus指标

            # v5.4: 记录协作模式统计
            if mode:
                self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
                collaboration_mode_total.labels(mode_name=mode).inc()  # Prometheus指标

            # v5.4: 记录查询意图分布
            if query_type:
                intent_distribution_total.labels(intent_type=query_type).inc()

            # v5.4: 记录用户满意度
            if satisfaction_score and 1 <= satisfaction_score <= 5:
                user_satisfaction_score.observe(satisfaction_score)

            # Business KPI tracking（v3.2: 细粒度解决率）
            now = time.time()
            if session_id is not None:
                is_first_turn = session_id not in self.session_turn_counts
                self.session_turn_counts[session_id] = (
                    self.session_turn_counts.get(session_id, 0) + 1
                )
                self.session_last_activity[session_id] = now
                if (
                    is_first_turn
                    and not escalated
                    and not cached
                    and resolution_status == "resolved"
                ):
                    self.total_single_turn_resolved += 1
                # 定期清理过期会话统计（每 100 次请求清理一次）
                if self.total_requests % CLEANUP_INTERVAL == 0:
                    self._cleanup_expired_sessions(now)
            if escalated:
                self.total_escalated += 1
            else:
                self.total_ai_handled += 1
            # v3.2: 细粒度解决状态统计
            if resolution_status and resolution_status in self.resolution_counts:
                self.resolution_counts[resolution_status] += 1
            # v5.0: 记录查询日志（用于热门问题和质量趋势）
            self._query_log.append((now, session_id or "", agent, mode, elapsed))

    async def record_feedback(self, resolved: bool, category: str = ""):
        """v3.6: 安全记录反馈（获取锁防止数据竞争）"""
        async with self._ensure_lock():
            if resolved:
                self.resolution_counts["resolved"] += 1
            else:
                self.resolution_counts["failed"] += 1
            # v5.0: 按分类统计反馈
            if category:
                if category not in self._feedback_by_category:
                    self._feedback_by_category[category] = {"positive": 0, "negative": 0}
                if resolved:
                    self._feedback_by_category[category]["positive"] += 1
                else:
                    self._feedback_by_category[category]["negative"] += 1

    def _cleanup_expired_sessions(self, now: float):
        """清理过期的会话统计，防止内存无限增长"""
        expired = [
            sid for sid, ts in self.session_last_activity.items() if now - ts > self._session_ttl
        ]
        for sid in expired:
            self.session_turn_counts.pop(sid, None)
            self.session_last_activity.pop(sid, None)
        if expired:
            logger.debug(f"[Metrics] 清理 {len(expired)} 个过期会话统计")

    # --- Section: Metric Queries (read path) ---
    async def update_business_metrics(self):
        """
        v5.4: 更新业务指标Gauge（解决率、升级率等）

        建议每60秒调用一次，或在关键事件后调用
        """
        async with self._ensure_lock():
            # 计算会话解决率
            total_sessions = len(self.session_turn_counts)
            if total_sessions > 0:
                resolved_count = self.resolution_counts.get("resolved", 0)
                resolution_rate = resolved_count / max(total_sessions, 1)
                session_resolution_rate.set(resolution_rate)

            # 计算人工升级率
            if self.total_requests > 0:
                esc_rate = self.total_escalated / self.total_requests
                escalation_rate.set(esc_rate)

            # 计算缓存命中率
            total_cache_ops = self.cache_hits + self.cache_misses
            if total_cache_ops > 0:
                cache_hit_rate = self.cache_hits / total_cache_ops
                business_cache_hit_rate.set(cache_hit_rate)

    # ===== v6.1: 新指标记录方法 =====

    async def record_cache_hit(self, level: int):
        """记录缓存命中层级（1=L1 MD5, 2=L2 Qdrant, 3=L3 Jaccard）"""
        if level == 1:
            cache_l1_hits_total.inc()
        elif level == 2:
            cache_l2_hits_total.inc()
        elif level == 3:
            cache_l3_hits_total.inc()
        async with self._ensure_lock():
            self.cache_hits += 1

    async def record_stream_ttfb(self, seconds: float):
        """记录流式首字响应时间"""
        await asyncio.to_thread(stream_ttfb_seconds.observe, seconds)

    async def record_trace_span(self):
        """记录 trace span"""
        trace_spans_total.inc()

    async def record_rag_query(self, recall: float | None = None):
        """记录 RAG 查询和召回率"""
        rag_queries_total.inc()
        if recall is not None:
            rag_recall_at_3.set(recall)

    async def record_scene_route(self, scene_name: str):
        """记录场景路由"""
        scene_routing_total.labels(scene_name=scene_name).inc()

    async def set_active_components(self, count: int):
        """设置活跃 DI 组件数"""
        active_components_total.set(count)

    async def get_stats(self) -> dict[str, Any]:
        async with self._ensure_lock():
            times = list(self.response_times)[-STATS_RECENT_COUNT:]  # 最近 N 次
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(
                    self.total_errors / max(self.total_requests, 1) * PERCENTAGE_MULTIPLIER, 1
                ),
                "avg_response_time": round(sum(times) / max(len(times), 1), 2),
                "p95_response_time": round(
                    sorted(times)[int(len(times) * P95_PERCENTILE)]
                    if len(times) >= P95_MIN_SAMPLES
                    else (max(times) if times else 0),
                    2,
                ),
                "cache_hit_rate": round(
                    self.cache_hits
                    / max(self.cache_hits + self.cache_misses, 1)
                    * PERCENTAGE_MULTIPLIER,
                    1,
                ),
                "agent_call_counts": dict(self.agent_call_counts),
                "mode_counts": dict(self.mode_counts),
                "sla": {
                    "target_min": RESPONSE_TIME_TARGET_MIN,
                    "target_max": RESPONSE_TIME_TARGET_MAX,
                    "violations_slow": self.sla_violations,
                    "violations_fast": self.sla_too_fast,
                    "violation_rate": round(
                        self.sla_violations / max(self.total_requests, 1) * PERCENTAGE_MULTIPLIER, 1
                    ),
                    "window_violation_rate": await self.get_sla_window_violation_rate(
                        _internal=True
                    ),
                },
            }

    async def get_sla_window_violation_rate(self, _internal: bool = False) -> float:
        """v3.2: 计算滑动窗口内的 SLA 违约率（v3.5: _internal 跳过锁，供已持锁的方法内部调用）"""
        if _internal:
            if not self._sla_window:
                return 0.0
            violations = sum(1 for t in self._sla_window if t > RESPONSE_TIME_TARGET_MAX)
            return round(violations / len(self._sla_window) * PERCENTAGE_MULTIPLIER, 1)
        async with self._ensure_lock():
            if not self._sla_window:
                return 0.0
            violations = sum(1 for t in self._sla_window if t > RESPONSE_TIME_TARGET_MAX)
            return round(violations / len(self._sla_window) * PERCENTAGE_MULTIPLIER, 1)

    async def get_kpi_stats(self) -> dict[str, Any]:
        async with self._ensure_lock():
            total_sessions = len(self.session_turn_counts)
            single_turn_sessions = sum(1 for v in self.session_turn_counts.values() if v == 1)
            first_resolution_rate = (
                single_turn_sessions / max(total_sessions, 1) * PERCENTAGE_MULTIPLIER
            )
            ai_handled_rate = (
                self.total_ai_handled / max(self.total_requests, 1) * PERCENTAGE_MULTIPLIER
            )

            # v3.2 口径：基于 Agent 信号的解决率（更准确）
            total_resolution = sum(self.resolution_counts.values())
            resolved = self.resolution_counts.get("resolved", 0)
            resolution_rate = resolved / max(total_resolution, 1) * PERCENTAGE_MULTIPLIER

            return {
                "first_resolution_rate": f"{first_resolution_rate:.1f}%",
                "first_resolution_detail": f"{single_turn_sessions}/{total_sessions}",
                # v3.2 增强指标
                "resolution_rate": f"{resolution_rate:.1f}%",
                "resolution_detail": {
                    "resolved": resolved,
                    "uncertain": self.resolution_counts.get("uncertain", 0),
                    "failed": self.resolution_counts.get("failed", 0),
                    "escalated": self.resolution_counts.get("escalated", 0),
                    "total": total_resolution,
                },
                "ai_handled_rate": f"{ai_handled_rate:.1f}%",
                "labor_savings_estimate": f"{ai_handled_rate:.1f}%",
                "total_ai_handled": self.total_ai_handled,
                "total_escalated": self.total_escalated,
                "total_single_turn_resolved": single_turn_sessions,
                "total_multi_turn": total_sessions - single_turn_sessions,
            }

    # --- Section: Monitoring Dashboard Queries ---
    async def get_quality_trends(self, days: int = 7) -> list[dict]:
        """最近 N 天质量评分趋势（基于真实查询日志）"""
        async with self._ensure_lock():
            from datetime import date, timedelta

            now = time.time()
            trends = []
            for i in range(days - 1, -1, -1):
                day_start = now - (i + 1) * 86400
                day_end = now - i * 86400
                day_entries = [e for e in self._query_log if day_start <= e[0] < day_end]
                if day_entries:
                    avg_time = sum(e[4] for e in day_entries) / len(day_entries)
                    score = max(0, min(100, 100 - avg_time * 2))
                else:
                    score = 0.0
                d = date.today() - timedelta(days=i)
                trends.append(
                    {
                        "date": d.isoformat(),
                        "avg_score": round(score, 1),
                        "total_queries": len(day_entries),
                    }
                )
            return trends

    async def get_hot_questions(self, limit: int = 10) -> list[dict]:
        """热门问题 TOP N（基于真实查询日志）"""
        async with self._ensure_lock():
            from collections import Counter

            # 按 agent 分类统计查询频次
            agent_counts: Counter = Counter()
            for entry in self._query_log:
                agent = entry[2] if len(entry) > 2 else ""
                if agent:
                    agent_counts[agent] += 1
            # 按 mode 分类
            mode_counts: Counter = Counter()
            for entry in self._query_log:
                mode = entry[3] if len(entry) > 3 else ""
                if mode:
                    mode_counts[mode] += 1
            # 返回 agent 维度的热门统计
            questions = []
            for agent, count in agent_counts.most_common(limit):
                questions.append({"agent": agent, "count": count})
            return questions

    async def get_satisfaction_stats(self) -> dict:
        """客户满意度统计（基于真实反馈数据）"""
        async with self._ensure_lock():
            total_positive = sum(c.get("positive", 0) for c in self._feedback_by_category.values())
            total_negative = sum(c.get("negative", 0) for c in self._feedback_by_category.values())
            total = total_positive + total_negative
            rate = round(total_positive / max(total, 1), 2)
            by_category = {}
            for cat, counts in self._feedback_by_category.items():
                cat_total = counts["positive"] + counts["negative"]
                by_category[cat] = {
                    "rate": round(counts["positive"] / max(cat_total, 1), 2),
                    "count": cat_total,
                }
            return {
                "overall_rate": rate,
                "total": total,
                "positive": total_positive,
                "negative": total_negative,
                "by_category": by_category,
            }

        # --- Section: Snapshot Persistence (optional, requires Redis) ---

    async def save_snapshot(self, redis_client=None) -> bool:
        """持久化指标快照到 Redis（可选，需 Redis 可用）"""
        if redis_client is None:
            return False
        try:
            snapshot = {
                "timestamp": time.time(),
                "stats": await self.get_stats(),
                "kpi": await self.get_kpi_stats(),
            }
            redis_client.set(
                "metrics:snapshot", json.dumps(snapshot, ensure_ascii=False), ex=METRICS_SNAPSHOT_TTL
            )
            redis_client.lpush("metrics:history", json.dumps(snapshot, ensure_ascii=False))
            redis_client.ltrim("metrics:history", 0, METRICS_HISTORY_MAX - 1)
            return True
        except Exception as e:
            logger.warning(f"[Metrics] 持久化快照失败: {e}")
            return False

    @staticmethod
    def load_snapshot(redis_client=None) -> dict[str, Any] | None:
        """从 Redis 加载最近的指标快照"""
        if redis_client is None:
            return None
        try:
            data = redis_client.get("metrics:snapshot")
            return json.loads(data) if data else None
        except Exception as e:
            logger.debug(f"[Metrics] 加载快照失败: {e}")
            return None


# ===== 模型熔断器 =====
class CircuitBreaker:
    """
    LLM 调用熔断器（v3.4: 状态转换原子化，防止多协程同时探测）
    连续失败达到阈值后进入 OPEN 状态，
    跳过 LLM 调用直接走降级路径（规则分类），恢复时间后进入 HALF_OPEN 尝试探测。
    三态：CLOSED（正常）→ OPEN（熔断）→ HALF_OPEN（探测）→ CLOSED
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_threshold: int = None, recovery_time: int = None):
        self.fail_threshold = (
            fail_threshold if fail_threshold is not None else CIRCUIT_BREAKER_FAIL_THRESHOLD
        )
        self.recovery_time = (
            recovery_time if recovery_time is not None else CIRCUIT_BREAKER_RECOVERY_TIME
        )
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.total_failures = 0
        self.total_successes = 0
        self._lock = asyncio.Lock()  # P1-2: 直接初始化，消除懒初始化竞态
        self._half_open_permits = 0  # H2: 控制 HALF_OPEN 探测次数

    def _ensure_lock(self):
        """P1-2: Lock 已在 __init__ 中初始化，直接返回"""
        return self._lock

    async def record_success(self):
        async with self._ensure_lock():
            self.consecutive_failures = 0
            self.total_successes += 1
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self._half_open_permits = 0
                logger.info("[CircuitBreaker] HALF_OPEN → CLOSED，LLM 恢复正常")

    async def record_failure(self):
        async with self._ensure_lock():
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                self._half_open_permits = 0
                logger.warning(
                    f"[CircuitBreaker] HALF_OPEN → OPEN，探测失败，重新熔断 {self.recovery_time}s"
                )
            elif self.consecutive_failures >= self.fail_threshold and self.state == self.CLOSED:
                self.state = self.OPEN
                logger.warning(
                    f"[CircuitBreaker] CLOSED → OPEN，连续 {self.consecutive_failures} 次 LLM 失败，"
                    f"熔断 {self.recovery_time}s，后续走降级路径"
                )

    async def should_allow(self) -> bool:
        """v3.4: 原子化状态检查+转换，防止多个协程同时进入 HALF_OPEN"""
        async with self._ensure_lock():
            if self.state == self.CLOSED:
                return True
            if self.state == self.OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_time:
                    self.state = self.HALF_OPEN
                    self._half_open_permits = 1
                    logger.info("[CircuitBreaker] OPEN → HALF_OPEN，尝试探测 LLM")
                    if self._half_open_permits > 0:
                        self._half_open_permits -= 1
                        return True
                    return False
                return False
            # HALF_OPEN: 仅允许一次探测
            if self._half_open_permits > 0:
                self._half_open_permits -= 1
                return True
            return False

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "fail_threshold": self.fail_threshold,
            "recovery_time": self.recovery_time,
        }


# ===== SLA 告警管理器 =====
class SLAAlertManager:
    """
    SLA 告警管理器（v5.4 - 分级告警 + 升级机制）

    核心功能：
    1. 基于滑动窗口检测SLA违约率
    2. 超过阈值时发布分级告警（warning/critical/emergency）
    3. 支持冷却机制避免告警风暴
    4. v5.4新增：告警升级检查（无人响应时自动升级）

    告警分级策略：
    - warning: 违约率 > 30%
    - critical: 违约率 > 60%
    - emergency: 违约率 > 90% 或 critical持续30分钟

    使用示例：
        >>> manager = SLAAlertManager(bus=message_bus)
        >>> alert = await manager.check_and_alert(metrics_collector)
        >>> if alert:
        ...     print(f"告警: {alert['message']}")
    """

    ALERT_HISTORY_MAX = 100  # 告警历史保留上限

    def __init__(self, bus=None):
        self.bus = bus
        self.alerts: list[dict[str, Any]] = []
        self.last_alert_time: dict[str, float] = {}

        # v5.4: 活动告警跟踪（用于升级检查）
        self.active_alerts: dict[str, dict[str, Any]] = {}

    async def check_and_alert(self, metrics: MetricsCollector) -> dict[str, Any] | None:
        """
        检查SLA违约率并发布告警

        Args:
            metrics: 指标收集器实例

        Returns:
            dict | None: 如果触发告警返回告警信息，否则返回None
        """
        window_rate = await metrics.get_sla_window_violation_rate()
        alert_key = "sla_violation_high"

        if window_rate > SLA_ALERT_THRESHOLD:
            last_time = self.last_alert_time.get(alert_key, 0)
            if time.time() - last_time < SLA_ALERT_COOLDOWN:
                return None

            # v5.4: 更精细的分级策略
            if window_rate > SLA_ALERT_THRESHOLD * 3:
                severity = "emergency"
            elif window_rate > SLA_ALERT_THRESHOLD * 2:
                severity = "critical"
            else:
                severity = "warning"

            alert = {
                "type": alert_key,
                "severity": severity,
                "message": f"SLA 违约率 {window_rate}% 超过阈值 {SLA_ALERT_THRESHOLD}%（最近 {SLA_ALERT_WINDOW} 次请求）",
                "window_rate": window_rate,
                "threshold": SLA_ALERT_THRESHOLD,
                "window_size": len(metrics._sla_window),
                "timestamp": time.time(),
            }

            self.alerts.append(alert)
            if len(self.alerts) > self.ALERT_HISTORY_MAX:
                self.alerts = self.alerts[-self.ALERT_HISTORY_MAX :]
            self.last_alert_time[alert_key] = time.time()

            # v5.4: 记录活动告警
            self.active_alerts[alert_key] = {
                "severity": severity,
                "timestamp": time.time(),
                "escalated": False,
            }

            logger.warning(f"[SLA-Alert] {alert['message']} (severity={severity})")

            if self.bus:
                try:
                    from core.message_bus import Message, MessageType

                    await self.bus.publish(
                        Message(
                            msg_type=MessageType.BROADCAST,
                            topic="alert.sla",
                            sender="sla_alert_manager",
                            payload=alert,
                        )
                    )
                except Exception as e:
                    logger.debug(f"[SLA-Alert] Bus 事件发布失败: {e}")

            # 分级通知路由
            try:
                from alerts.notifier import alert_notifier

                await alert_notifier.send_alert(
                    title=f"SLA 告警 [{severity.upper()}]",
                    content=alert["message"],
                    severity=severity,
                )
            except Exception as e:
                logger.debug(f"[SLA-Alert] 通知发送失败: {e}")

            return alert
        return None

    async def check_and_upgrade(self):
        """
        v5.4: 检查并升级活动告警

        调用此方法定期检查是否有告警需要升级
        建议在后台任务中每5分钟调用一次

        Returns:
            list: 已升级的告警列表
        """
        upgraded = []
        now = time.time()

        for alert_key, alert_info in list(self.active_alerts.items()):
            elapsed = now - alert_info["timestamp"]
            severity = alert_info["severity"]

            # critical → emergency 升级（30分钟未解决）
            if severity == "critical" and elapsed > 1800:
                if not alert_info["escalated"]:
                    logger.warning(f"[SLA-Alert] 告警升级: {alert_key} critical → emergency")

                    # 发送升级通知
                    try:
                        from alerts.notifier import alert_notifier

                        await alert_notifier.send_alert(
                            title="[升级] SLA 告警",
                            content=f"SLA违约告警已持续{elapsed//60:.0f}分钟未解决，已升级为emergency级别",
                            severity="emergency"
                        )

                        alert_info["escalated"] = True
                        alert_info["severity"] = "emergency"
                        upgraded.append(alert_key)
                    except Exception as e:
                        logger.error(f"[SLA-Alert] 升级通知失败: {e}")

        return upgraded

    def get_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近的告警历史"""
        return self.alerts[-limit:]

    def get_active_alerts(self) -> dict[str, dict[str, Any]]:
        """v5.4: 获取当前活动告警状态"""
        return self.active_alerts.copy()
