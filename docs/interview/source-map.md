# Interview Source Map — 面试问题 → 源码 → 测试 → 证据

> **这份表是入口。** 面试官问任何一个问题，从这里出发，可以在两步之内落到
> 真实源码、真实测试和真实证据上。
>
> 每一行的 `Evidence` 列只写**当前真实存在的证据等级**，不写 aspiration。
> 证据等级的定义见 [production-evidence.md](../evaluation/production-evidence.md)。
>
> **禁止表述**（任何材料都不得出现，除非有对应 artifact）：
> 生产级稳定 / 生产验证通过 / 高并发生产验证 / Exactly-once / 零重复 /
> 100% crash recovery / 真实 ERP 已验证 / Kubernetes production ready /
> 649-query RAG verified
>
> 允许表述：IMPLEMENTED / LOCALLY VERIFIED / CI VERIFIED / CONTROLLED STAGING

---

## 1. 分布式 Runtime

| 面试问题 | 核心源码 | 测试 | 文档 | Evidence |
|---|---|---|---|---|
| Agent 执行如何持久化 | `runtime/executor.py::execute_run`、`runtime/run_service.py`、`db/models.py::AgentRun` | `tests/integration/runtime/test_run_semantics.py` | [runtime-deep-dive.md](runtime-deep-dive.md) | CI VERIFIED（真实 PG） |
| 为什么 AgentRun 表是唯一真相源 | `runtime/statuses.py`（状态机定义）、`runtime/run_service.py`（原子条件更新） | `tests/unit/test_distributed_runtime.py` | [runtime-deep-dive.md](runtime-deep-dive.md) §状态机 | IMPLEMENTED + CI VERIFIED |
| checkpoint 和 AgentRun 有什么区别 | `core/checkpointer.py`（LangGraph 状态）、`db/models.py::AgentRun`（业务状态） | `tests/integration/runtime/test_cross_process_checkpoint.py` | [runtime-deep-dive.md](runtime-deep-dive.md) §checkpoint vs AgentRun | CI VERIFIED |
| worker 崩溃如何恢复 | `runtime/bootstrap.py::invoke_graph_with_resume`、`runtime/celery_app.py`（acks_late / reject_on_worker_lost） | `tests/integration/runtime/test_worker_checkpoint_recovery.py`、`scripts/probe_crash_kill_gate.py` | [runtime-deep-dive.md](runtime-deep-dive.md) §崩溃恢复 | CI VERIFIED（真实 SIGKILL + PG checkpoint） |
| **那个 flaky test 的根因是什么** | `tests/integration/runtime/test_worker_checkpoint_recovery.py::first_superstep_durable` | `tests/unit/test_crash_recovery_kill_gate.py`（12 例）、`artifacts/flake-investigation/killgate-*/` | [runtime-deep-dive.md](runtime-deep-dive.md) §flaky 根因 | CI VERIFIED（270 cycles 探针 + 端到端复现） |
| 失败时如何取证 | `tests/integration/runtime/crash_diagnostics.py`、`conftest.py::pytest_runtest_makereport` | `artifacts/runtime-diagnostics-selftest/` | [runtime-deep-dive.md](runtime-deep-dive.md) §可观测性 | LOCALLY VERIFIED |
| 为什么是 at-least-once 不是 exactly-once | `runtime/celery_app.py` 模块 docstring、`docs/design/distributed-agent-runtime.md` | `tests/integration/runtime/test_tool_idempotency.py` | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §1 | IMPLEMENTED |
| 怎么防重复消费 | `runtime/repository.py::mark_running`（lease 原子领取）、`runtime/thread_lock.py` | `tests/integration/runtime/test_run_semantics.py`（lease 抢占） | [runtime-deep-dive.md](runtime-deep-dive.md) §租约 | CI VERIFIED |
| **怎么防重复退款**（最常问） | `runtime/side_effects.py::SideEffectStore.claim`、`tools/tool_registry.py::_idempotent_operation` | `tests/integration/runtime/test_tool_idempotency.py`、`test_side_effect_claim_concurrency.py`（8 并发只 1 执行）、`test_tool_idempotency_metric.py` | [runtime-deep-dive.md](runtime-deep-dive.md) §副作用 ledger | CI VERIFIED（真实并发 + worker kill） |
| 为什么不用分布式事务 | — | — | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §2 | 设计决策 |
| 幂等键怎么构造 | `runtime/side_effects.py::default_operation_key`、`runtime/executor.py`（`run_id:tool_call_id`） | `tests/unit/test_distributed_runtime.py` | [runtime-deep-dive.md](runtime-deep-dive.md) §幂等键 | CI VERIFIED |
| 重试与退避怎么调度 | `runtime/retry.py`、`runtime/executor.py::_handle_failure` | `tests/integration/runtime/test_run_semantics.py`（transient ×2 → success） | [runtime-deep-dive.md](runtime-deep-dive.md) §retry | CI VERIFIED |
| DLQ 是什么、怎么重放 | `db/models.py::AgentDeadLetter`、`scripts/replay_dead_run.py` | `tests/integration/runtime/test_run_semantics.py`（DLQ + replay 保持 run_id） | [runtime-deep-dive.md](runtime-deep-dive.md) §DLQ | CI VERIFIED |
| 分布式锁解决了什么 | `runtime/thread_lock.py`（Redis owner token + TTL + Lua CAS） | `tests/integration/runtime/test_run_semantics.py`（同 thread 串行 / 跨 thread 并发） | [runtime-deep-dive.md](runtime-deep-dive.md) §锁 | CI VERIFIED（真实 Redis） |
| run 事件流是真相源吗 | `runtime/events.py` | `tests/integration/runtime/test_event_delivery_semantics.py`（明确断言"不是真相源"） | [runtime-deep-dive.md](runtime-deep-dive.md) §run events | CI VERIFIED |
| 多 worker 一致性怎么保证 | `core/config.py::validate_distributed_runtime_settings` | `tests/unit/test_execution_mode_contract.py` | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §Redis 不是 canonical | CI VERIFIED（配置契约测试） |
| `/api/chat` 和 `/api/runs` 有什么区别 | `api/app.py::_run_graph`、`api/routes/runs.py`、`runtime/dispatch.py` | `tests/integration/runtime/test_queue_worker_decoupling.py` | [architecture-walkthrough.md](architecture-walkthrough.md) §4 | CI VERIFIED |

---

## 2. HITL（人工审批治理）

| 面试问题 | 核心源码 | 测试 | 文档 | Evidence |
|---|---|---|---|---|
| HITL 是怎么实现的 | `core/hitl/risk.py`、`core/hitl/gate.py`、`core/hitl/approval_service.py`、`api/routes/approvals.py` | `tests/integration/runtime/test_hitl_approval_flow.py`、`test_hitl_langgraph_interrupt.py`、`tests/unit/test_hitl_approval.py` | [hitl-deep-dive.md](hitl-deep-dive.md) | CI VERIFIED |
| 风险怎么分级 | `core/hitl/risk.py::classify_risk`（工具声明 > 名称 allowlist > 金额阈值 > 默认 LOW） | `tests/unit/test_hitl_risk.py`、`test_hitl_gate.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §分级 | CI VERIFIED |
| 拦截发生在副作用之前吗 | `tools/tool_registry.py::_idempotent_operation`（先 claim 再执行）、`core/hitl/gate.py::should_propose_approval` | `tests/integration/runtime/test_hitl_approval_flow.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §执行前拦截 | CI VERIFIED |
| WAITING_APPROVAL 为什么不消耗 retry | `runtime/statuses.py`（`APPROVAL_RESUMABLE_STATUSES` 与 `EXECUTABLE_STATUSES` 分离）、`runtime/run_service.py::mark_resumed_running` | `tests/unit/test_hitl_run_status.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §attempt 语义 | CI VERIFIED |
| **approval ≠ idempotency** | `core/hitl/approval_service.py`（决策幂等）vs `runtime/side_effects.py`（执行幂等） | `tests/unit/test_hitl_run_status.py`、`test_tool_idempotency.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §两道防线 | CI VERIFIED |
| TTL 到了怎么办 | `core/hitl/approval_service.py::is_expired`（过期按拒绝，**绝不默认放行**） | `tests/integration/runtime/test_hitl_approval_flow.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §TTL fail closed | CI VERIFIED |
| 职责分离怎么保证 | `core/hitl/approval_service.py::_guard_reviewer`（`reviewer_id != user_id`） | `tests/integration/runtime/test_hitl_approval_flow.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §separation of duties | CI VERIFIED |
| RBAC 在哪一层 | `api/routes/approvals.py`（API 层 RBAC） | `tests/unit/test_hitl_api.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §RBAC | CI VERIFIED |
| 审批后如何恢复（会不会执行两次） | `runtime/bootstrap.py::invoke_graph_with_resume`（`Command(resume=...)`，实测 langgraph 1.2.12 语义） | `tests/integration/runtime/test_hitl_langgraph_interrupt.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §恢复语义 | CI VERIFIED |
| `/api/chat` 快路径在 HITL 边界内吗 | `api/app.py::_run_graph`（inline，不落 AgentRun） | `tests/integration/runtime/test_queue_worker_decoupling.py` | [hitl-deep-dive.md](hitl-deep-dive.md) §不在边界内 | CI VERIFIED |

---

## 3. RAG

| 面试问题 | 核心源码 | 测试 | 文档 | Evidence |
|---|---|---|---|---|
| RAG 链路怎么走 | `rag/qdrant_knowledge_base.py::retrieve`、`rag/retrieval_contract.py`（7 阶段） | `tests/unit/test_p1_01_retrieval_contract.py`、`test_llm_rag_coverage.py` | [rag-deep-dive.md](rag-deep-dive.md) | LOCALLY VERIFIED |
| 为什么不是单向量检索 | `rag/qdrant_knowledge_base.py`（BM25 通道 + `rrf_fusion`） | `tests/unit/test_rag_eval_harness.py` | [rag-deep-dive.md](rag-deep-dive.md) §为什么要混合 | LOCALLY VERIFIED |
| 什么是 RRF，为什么用它 | `rag/qdrant_knowledge_base.py::rrf_fusion`（k=60，标准值） | `tests/unit/test_p1_01_retrieval_contract.py` | [rag-deep-dive.md](rag-deep-dive.md) §融合 | LOCALLY VERIFIED |
| 为什么需要 reranker | `rag/reranker.py`、`STAGE_RERANK` | `tests/unit/test_rag_reranker.py` | [rag-deep-dive.md](rag-deep-dive.md) §精排 | LOCALLY VERIFIED |
| embedding 不可用会怎样 | `rag/embedding_status.py`、`rag/qdrant_knowledge_base.py`（fail-closed，绝不用假向量） | `tests/unit/test_cache_embedding_fail_closed.py` | [rag-deep-dive.md](rag-deep-dive.md) §降级 | CI VERIFIED（PR #12） |
| **RAG 如何评测** | `scripts/evaluate_rag.py`、`tests/eval/rag_benchmark.json`（649 条） | `scripts/eval_contract.py` | [rag-deep-dive.md](rag-deep-dive.md) §评测口径 | **NOT_VERIFIED**（provider 401 阻塞） |
| 当前 Recall@K / MRR / NDCG 是多少 | — | — | [rag-deep-dive.md](rag-deep-dive.md) §指标状态 | **NOT_MEASURED** — 不得引用任何数字 |
| 负样本怎么构造 | `tests/eval/golden/expected_doc_ids.json`（正样本标注；负样本 = 语料内非 gold 文档） | `scripts/eval_contract.py` | [rag-deep-dive.md](rag-deep-dive.md) §负样本 | IMPLEMENTED |
| 为什么 Recall@K ≠ 最终回答质量 | — | — | [rag-deep-dive.md](rag-deep-dive.md) §指标分层 | 设计决策 |

---

## 4. 可观测性 / 追踪

| 面试问题 | 核心源码 | 测试 | 文档 | Evidence |
|---|---|---|---|---|
| 一次请求怎么端到端追踪 | `core/tracing.py`（基础设施）、`core/telemetry.py`（应用语义） | `tests/unit/test_telemetry.py`（23 例） | [architecture-walkthrough.md](architecture-walkthrough.md) §tracing | LOCALLY VERIFIED |
| span 拓扑是什么 | `runtime/executor.py`、`rag/qdrant_knowledge_base.py`、`llm/client.py`、`tools/tool_registry.py` | `tests/unit/test_telemetry.py` | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §观测边界 | LOCALLY VERIFIED |
| trace 里会不会泄露用户数据 | `core/telemetry.py::ALLOWED_ATTRIBUTES` + `FORBIDDEN_SUBSTRINGS` | `tests/unit/test_telemetry.py::test_content_attributes_are_dropped` | — | CI VERIFIED（白名单测试） |
| tracing 坏了会不会影响业务 | `core/tracing.py::safe_span`（只依赖 contextlib 的最后降级） | `tests/unit/test_telemetry.py::test_telemetry_module_never_raises_when_otel_sdk_is_absent` | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §观测不是单点 | CI VERIFIED |
| HITL 前后两段 trace 怎么关联 | `core/telemetry.py` 模块 docstring（`run_id` / `approval_id`，**不**承诺单 span） | — | [hitl-deep-dive.md](hitl-deep-dive.md) §trace 边界 | IMPLEMENTED（诚实边界） |

---

## 5. 文档一致性 / 工程纪律

| 面试问题 | 核心源码 | 测试 | 文档 | Evidence |
|---|---|---|---|---|
| 怎么防止文档漂移 | `scripts/audit_doc_consistency.py`、`scripts/project_facts.py`、`scripts/generate_openapi.py` | `tests/unit/test_doc_consistency.py` | [current-state.md](../reference/current-state.md) | CI VERIFIED |
| 证据等级怎么定义 | — | — | [production-evidence.md](../evaluation/production-evidence.md) | CURRENT |
| 当前 runtime 事实 | `core/config.py::VERSION` | `scripts/project_facts.py --check` | [current-state.md](../reference/current-state.md) | CI VERIFIED |

---

## 6. 明确**不是**当前能力（避免面试被反问）

| 主题 | 当前状态 | 权威文档 |
|---|---|---|
| 生产环境验证 | **NOT_VERIFIED** | [production-evidence.md](../evaluation/production-evidence.md) |
| 真实 ERP 写操作 | **NOT_MEASURED**（Mock 为主） | [production-evidence.md](../evaluation/production-evidence.md) |
| 649-query RAG 指标 | **NOT_MEASURED**（provider 401） | [rag-evaluation.md](../reference/rag-evaluation.md) |
| Kubernetes 生产就绪 | **未进入主线**（见 §MCP / K8s 处理） | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §为什么没有直接上 K8s |
| MCP 工具适配 | **已进主线，默认关闭**（`MCP_ENABLED=false`；read-only-first，只注册显式 `low` 的 server 工具）。端到端契约取证 `NOT_VERIFIED` | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §为什么 MCP 以「默认关闭 + 只读优先」的方式进主线 |
| 多租户 / SAML / OIDC / 完整 IAM | **有意不做** | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §为什么没有 multi-tenant |
| exactly-once 投递 | **有意不做**（at-least-once + 三层幂等） | [failure-and-tradeoffs.md](failure-and-tradeoffs.md) §1 |

---

## 7. 复现本表所有验证命令

```bash
# 单元测试（不需要外部基础设施）
pytest tests/unit -q

# 真实基础设施 runtime 验收（需要真实 PostgreSQL + Redis）
make runtime-e2e

# 崩溃恢复 flaky 的定点取证（本次根因定位用）
python3 scripts/probe_crash_kill_gate.py --cycles 150 --tail-on-hit

# 崩溃恢复重复验证（50 次，产出 summary.json）
python3 scripts/repeat_crash_recovery_test.py --runs 50

# 混沌验收
make runtime-chaos
make runtime-verify

# 文档一致性守卫
make audit-docs
make openapi-check
python3 scripts/project_facts.py --check docs/reference/current-state.md
```

**RAG 649 正式评测当前被 provider 认证阻塞**（HTTP 401 `"Token is invalid."`），
不要在没有解除阻塞的情况下引用任何 RAG 指标。见 Issue #7。