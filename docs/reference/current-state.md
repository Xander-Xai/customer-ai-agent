# 当前事实入口 (Current State)

> 本文件是当前 runtime 事实的唯一定位入口（Current entry point）。
> 它刻意**不硬编码 Git HEAD**：HEAD 请用 `git rev-parse HEAD` 获取。
> 带日期的历史审计报告（如 docs/reports/plans/**）只是当时的快照，不等于当前事实。
>
> 每个数字的获取方式都标注了命令；不要从旧文档复制这些数字。

## Runtime facts (generated)

以下值由 `python3 scripts/project_facts.py` 在当前 checkout 动态生成（以命令当前输出为准，不绑定验证日期；`.env` 本地覆盖不改变此处的 runtime fallback facts）：

- Runtime version (`core/config.py::VERSION`): **`6.3`**
- Default LLM (`core/config.py::OPENAI_MODEL`): **`Qwen/Qwen3-8B`**
- Default LLM base URL: `https://api.siliconflow.cn/v1`（provider: `siliconflow`）
- Default embedding (`core/config.py::EMBEDDING_MODEL`): **`BAAI/bge-large-zh-v1.5`**（1024 维）
- Default reranker (`core/config.py::RERANKER_MODEL`): **`BAAI/bge-reranker-v2-m3`**
- Vector DB (`core/config.py::VECTOR_DB_MODE`): `qdrant_only`（ChromaDB 已移除）
- Hybrid retrieval (`core/config.py::HYBRID_SEARCH_ENABLED`): `true`（向量 + BM25 + RRF 融合）
- Agent roles (`core/container.py::_init_agents`): **9** 个运行时角色
  （7 领域 Agent + ReActAgent + ResponseAgent；BaseAgent 是抽象基类、
  ResponseEvaluator 是质量评估器，两者不计入运行时角色）
- OpenAPI HTTP paths (`app.openapi()["paths"]`): **`62`**（`docs/openapi.json` 快照；
  数字随 approval endpoints 等新增而变，以本命令输出为准）
- RAG benchmark queries (`tests/eval/rag_benchmark.json` metadata): **`649`**

## RAG evaluation / evidence state

- **当前 649-query 正式指标（Hit@K / Recall@K / Precision@K / NDCG@K / MRR@K）：NOT_VERIFIED。**
  在 preflight 通过并产生正式 artifact 之前，不得把任何百分比写成当前事实。
- Evidence harness 支持 **4 个检索配置**（canonical 实验名以
  [docs/reference/rag-evaluation.md](rag-evaluation.md) 为准）：
  `vector_only` / `bm25_only` / `hybrid_no_rerank` / `hybrid_rerank`。
- Metric family：Hit@K、Recall@K、Precision@K、NDCG@K、MRR@K（multi-K：1/3/5/8）。
- Evaluation populations（全部运行时动态计算，禁止硬编码分母）：
  `all_queries`（主口径，end-to-end）/ `retrieval_eligible` / `full_gold_covered`。
- **最新已提交的 preflight evidence**：
  `artifacts/evaluation/rag-649/preflight-20261002T194209Z/report.json`
  （`schema_version: rag-eval-evidence/v2`，`timestamp 2026-10-02T19:42:09Z`，
  `status: BLOCKED`）。blocker 语义按 v2 结构化记录，**必须分开表述**：
  - `primary_blocker: EMBEDDING_PROVIDER_AUTH`（embedding 探针 HTTP 401，
    `blocks_corpus_import: true`）是**根因**；
  - `VECTOR_INDEX_EMPTY`（评测集合 0 points）是 **downstream 症状**，
    `caused_by: EMBEDDING_PROVIDER_AUTH`；因 BM25 索引由 Qdrant 重建，
    连 `bm25_only` 也被它阻塞；
  - `RERANKER_PROVIDER_AUTH`（reranker 探针 HTTP 401，`silent_fallback`）是
    **`blocking: false`**，只阻塞 `hybrid_rerank`，**不得**据此声称其它实验
    也被 reranker 阻塞。
  该 artifact 自述 `formal evaluation not run; no metrics generated`——
  `declared_queries: 649` / `executed_queries: 649` 是 preflight 的探针计数，
  **不是** 649-query 正式评测完成，正式指标仍为 `NOT_VERIFIED`。
  上一版 artifact（`preflight-20260929T191128Z`，v1 schema，顶层
  `status: BLOCKED_VECTOR_INDEX`）作为历史记录原样保留，不回填。
- 详细流程（import → preflight → smoke → formal）、artifact schema、
  blocker 语义与评测状态：[docs/reference/rag-evaluation.md](rag-evaluation.md)。
  canonical 命令链：`make rag-eval-import` → `make rag-eval-649-preflight` →
  `make rag-eval-649-smoke`（冒烟，非正式证据） → `make rag-eval-649`
  （`make eval-rag` 为其兼容 alias）。该文档在此方面内容为 **CURRENT**
  （随评测实现同步），其历史小节单独标注。

## 验证命令（不要复制数字，重新执行）

```bash
# runtime 事实（版本/模型/路径数/benchmark 数/Agent 角色数）
python3 scripts/project_facts.py

# OpenAPI 快照一致性（docs/openapi.json 与 app.openapi()）
python3 scripts/generate_openapi.py --check

# RAG benchmark metadata 一致性
python3 - <<'PY'
import json
d = json.load(open("tests/eval/rag_benchmark.json", encoding="utf-8"))
assert d["metadata"]["total_queries"] == len(d["queries"])
print("RAG benchmark queries:", len(d["queries"]))
PY

# 当前 pytest 收集数（测试数量不写入任何文档，以此命令为准）
pytest --collect-only -q

# 当前前端测试
npm test

# 文档一致性审计
python3 scripts/audit_doc_consistency.py
```

## 稳定的架构事实

- 四层状态机：缓存检查 → 意图路由（LLM ∥ 规则并行）→ 协作模式（Sequential /
  Parallel / Consultation / Hierarchical / ReAct）→ 响应后处理。
  入口：`core/graph_builder.py::build_graph`。
- **Response Cache**（`cache/response_cache.py` + `cache/cache_policy.py`）:
  L1 Redis 精确 → L2 Qdrant 语义 → L3 Jaccard 回退；scope/version/ttl 由
  CachePolicy 统一决定。
- **Tool Result Context Engineering**（`core/tool_result_*.py`）是独立机制：
  确定性压缩、Top-K/token 预算、历史 compaction、专用 compressor、可选
  offload/store/recovery、scope-safe exact reuse cache、可选 semantic summary。
  它不是 Response Cache 的一部分；Session Memory（会话窗口/摘要）是第三种独立概念。
- **LangGraph Checkpoint**（`core/checkpointer.py`）是第四种独立概念：图状态快照，
  键为 `thread_id`（当前实现 `thread_id == session_id`），支持断点续传。
  `LANGGRAPH_CHECKPOINT_BACKEND` 选择 `memory`（开发/测试，进程内
  `MemorySaver`）或 `postgres`（官方 `langgraph-checkpoint-postgres` 的
  `AsyncPostgresSaver`，跨 worker/副本共享、重启可恢复；留空时开发→memory、
  生产→postgres）。checkpoint 表由官方 saver 自管，不与业务 SQLAlchemy Base 耦合。
  生产初始化失败 fail closed，不静默回退 MemorySaver。
  四个机制（Session Memory / LangGraph Checkpoint / Response Cache /
  Tool Result Store）不是同一个概念，禁止互相替代或合并叙述。
- RAG 链路：rewrite/filter → vector + BM25 → retrieval contract
  （`rag/retrieval_contract.py`）→ RRF 融合 → rerank（`rag/reranker.py`）→ context。
  BM25 lifecycle: `rag/bm25_lifecycle.py`；确定性 point ID 与迁移：`rag/point_id.py`、
  `rag/point_id_migration.py`。
- **MCP 外部工具接入（`tools/mcp_adapter.py`）默认关闭**：`MCP_ENABLED=false`。
  它是 Function Calling 的**传输层扩展**，不是替代品 —— native 工具（ERP / RAG /
  系统内建）继续走进程内注册表，MCP 工具叠加进**同一个** `ToolRegistry`，
  同名时跳过、绝不覆盖 native。
  - 安全立场：MCP **不构成新的安全边界**。外部 server 是不可信输入，风险等级
    **沿用** `core.hitl.risk.RiskLevel`（low/medium/high），不另立词汇表。
  - `MCP_SERVERS` 是 JSON 数组 allowlist（空 = 不允许任何 server）；
    `allowed_tools` 为空同样等于不允许任何工具。
  - **read-only-first**：只有显式 `risk_level: "low"` 的 server 的工具才注册。
    缺失 / 非法（含历史词汇 `read` / `write`）**只向上**收敛到 `HIGH`，
    绝不 fail-open。`medium` / `high` 一律不注册。
  - **不采信 server 自述的 `annotations`** 来决定风险等级。
  - 命名空间 `mcp__{server}__{tool}`；server 名禁止含 `__`；超长名截断补 sha256。
  - 启动期结构校验：`core/config.py::validate_mcp_settings`（fail closed）。
    `MCP_FAIL_CLOSED=true` 时初始化失败阻止启动，而不是静默降级为 native-only。
  - **边界（不得越界宣称）**：写操作 MCP 工具**未接入**（缺幂等 ledger + 人工审批
    两道防线）；RBAC 是**请求级**而非 per-tool，不能声称"MCP 工具经过了 RBAC"；
    响应侧结果大小当前**不设上限**（只限请求 payload 字节），属已知缺口；
    官方 `mcp` SDK 是 `requirements-optional.txt` 里的**可选**依赖，延迟 import。
  - **Evidence**：`IMPLEMENTED`，仅**纯函数契约**经
    `tests/unit/test_mcp_adapter.py` 运行验证。跨进程 / 传输 / 策略 / 时序的
    **端到端契约取证当前为 `NOT_VERIFIED`**，在它落地并跑出真实结果前不得声称
    MCP 端到端可用。设计取舍详见
    [docs/interview/failure-and-tradeoffs.md](../interview/failure-and-tradeoffs.md) §7。
- Embedding 通过 HTTP API 计算（`rag/api_embedding.py`），应用侧计算、Qdrant 只做存储检索。
- LLM 客户端：`llm/client.py`（指数退避重试 + 熔断 + FC + SSE 流式 + 连接池）；
  降级兜底 `llm/rule_based_llm.py`。
- **应用语义 tracing**（`core/telemetry.py` + 基础设施 `core/tracing.py`）同样是独立
  机制，与 Session Memory / LangGraph Checkpoint / Response Cache / Tool Result Store
  都不是同一个概念。span 语义已接线且本地验证：
  `csai.agent.execute` / `csai.agent.execute.resume`（`runtime/executor.py`）、
  `csai.rag.retrieve` + `rag.stage.*` event（`rag/qdrant_knowledge_base.py`）、
  `csai.llm.chat_completion`（`llm/client.py`）、`csai.tool.execute`
  （`tools/tool_registry.py`）；属性白名单 + 敏感词过滤，任何 OTel 失败都降级为
  no-op 且不改变业务语义。**四个层级必须分开表述**：
  - 应用语义 tracing：**IMPLEMENTED / LOCALLY VERIFIED**（`tests/unit/test_telemetry.py`）；
  - 真实 OTLP Collector 传输链路（SDK → `OTLPSpanExporter` → 网络 → 真实
    Collector）：**LOCALLY VERIFIED**（`make otel-collector-smoke`，真实
    `otel/opentelemetry-collector:0.162.0`，OTLP gRPC；证据
    `artifacts/observability/otel-collector-20261003T040014Z/report.json`，
    schema `otel-collector-evidence/v1`，`status: VERIFIED_LOCAL`，5 个 semantic span
    全部到达 Collector）；
  - 持久化 / 可查询 trace 后端（Jaeger / Langfuse / Tempo 等）：`NOT_VERIFIED`
    ——被验证的 Collector 只有 `debug` exporter，不存储、无 retention、无查询 UI、
    无 dashboard。「Collector 收到了 trace」**不等于**「有 trace 后端」；
  - 生产 trace 传播 / 真实流量：`NOT_VERIFIED`——不得写成 "production-ready
    tracing" / "生产已验证的可观测性"。`OPENTELEMETRY_ENABLED` / `OTEL_ENABLED`
    在 `.env.example` 中仍默认 `false`。
  Collector 那一腿**不覆盖**真实 Agent → RAG → LLM → tool 全链路：smoke 直接用
  `core.telemetry.span` 发 span，避免为此拉起真实 LLM / Qdrant / ERP / Celery；
  call-site 接线由 `tests/unit/test_telemetry.py` 单独覆盖。
  两个边界随声明一起传播：审批前后两段**不承诺**是同一个 span（用 `run_id` /
  `approval_id` 关联）；属性是白名单而非采样（raw prompt、用户原文、召回文档、
  工具参数、PII、凭据一律不进 trace）。详见
  [docs/evaluation/production-evidence.md](../evaluation/production-evidence.md)。

## 分布式 Agent Runtime（异步 Run + Celery Worker）

- **Hybrid Architecture**：实时快路径 `POST /api/chat` / `/api/chat/stream`
  （FastAPI → LangGraph → SSE）保持不变；长任务走异步路径
  `POST /api/runs`（创建 Run + 入队，立即返回）→ Celery worker →
  `GET /api/runs/{run_id}` polling。见 `runtime/`、`api/routes/runs.py`。
- **三个 ID 严格区分**：
  - `thread_id`：对话级 ID（当前 `thread_id == session_id == LangGraph thread`），
    同一多轮会话复用；
  - `run_id`（`agent_runs.id`）：单轮 Graph 执行 ID，每次请求唯一；
  - `task_id`：队列消息 / Worker 执行 ID（Celery task id）。
  禁止"每个请求新建 thread_id"。
- **业务状态真相源**：数据库 `agent_runs` 表（canonical）。**完整合法迁移集合以
  `runtime/statuses.py::ALLOWED_TRANSITIONS` 为唯一真相源**，本节只是它的可读
  投影（不允许在别处再写第二份状态机）：

  ```text
  PENDING ──► QUEUED ──► RUNNING ──┬──► SUCCEEDED              (终态)
                    │              ├──► FAILED                 (终态，permanent error，不重试)
                    │              ├──► RETRYING ──► RUNNING   (transient error，退避后重试)
                    │              ├──► WAITING_APPROVAL ──► RUNNING  (人工审批决策后恢复)
                    │              └──► DEAD_LETTER            (终态，retry 用尽)
                    └──► CANCELLED                            (终态，用户/管理员取消)

  任意未终态 ──► CANCELLED（escape path）
  WAITING_APPROVAL ──► DEAD_LETTER / CANCELLED（审批被永久搁置时的逃生口）
  ```

  - `WAITING_APPROVAL` 是**非终态**，且**不在** `EXECUTABLE_STATUSES`
    （`{QUEUED, RETRYING}`）里：等待审批的 run 只能由审批 API 显式投递恢复，
    通用队列轮询不会把它当成待办反复捞起（否则无人处理的审批会变成忙循环）。
  - `WAITING_APPROVAL → RUNNING` 由 `RunService.mark_resumed_running()` 执行且
    **不递增 attempt**——等待人不是失败，不该消耗 `AGENT_RUN_MAX_ATTEMPTS`。
  - 刻意**没有** `WAITING_APPROVAL → QUEUED` 这条边（原因同上：改回 QUEUED 会走
    `mark_running`，而它每次领取都 `attempt + 1`）。
  - 终态：`SUCCEEDED | FAILED | DEAD_LETTER | CANCELLED`。Celery result backend
    **不是**真相源（`task_ignore_result=True`）。
- **Human-in-the-loop（高风险副作用治理）**：HIGH 风险工具副作用在**执行前**被拦下
  （Agent 工具循环只把它摘进 `pending_actions`，并未执行），durable 审批记录落
  `human_approvals` 表（`alembic 006`），图在 `interrupt()` 处挂起、checkpoint
  落库，run 转入上述 `WAITING_APPROVAL`；人工 approve/edit/reject（仅
  admin/supervisor，且 `reviewer != requester`）后由 worker 以
  `Command(resume=...)` 恢复，执行仍经 side-effect ledger
  （`operation_key = run_id:approval:{approval_id}`）。实现见 `core/hitl/`、
  `api/routes/approvals.py`、
  [design/human-in-the-loop.md](../design/human-in-the-loop.md)。
  **边界**：`HITL_ENABLED` 默认 `false`；`/api/chat` 快路径无 run 上下文，
  明确**不在**该治理边界内（不得宣称其受 durable HITL 保护）；
  **真实 ERP 写操作 NOT_VERIFIED**（无企业 staging，副作用验证走确定性 staging
  工具，见该文档 §9）。
- **取消（协作式）**：`POST /api/runs/{run_id}/cancel` 立即置 `CANCELLED`。
  未开始的 run 不会再被执行；**已进入 RUNNING 的 run 不会被强行中断**，
  调用方需轮询确认终态。
- **Checkpoint**：生产 `postgres`（`AsyncPostgresSaver`）跨 worker/副本共享，
  见上；API 快路径与 worker 异步路径共享同一 checkpoint 后端。
- **Thread lock（API 执行边界 + worker 共用）**：Redis
  `agent:thread-lock:{thread_id}`（owner token + TTL + 原子 compare-and-delete 释放）；
  同一 thread 串行，不同 thread 并发。REST/SSE/WS/multimodal 统一经
  `api/app.py::_run_graph` 获取锁；拿不到锁返回 `THREAD_BUSY`（HTTP 409 /
  SSE/WS 错误帧）。实现：`core/concurrency/distributed_lock.py`（复用
  `runtime/thread_lock.py`）。开发/测试用进程内锁，不依赖 Redis。
- **生产 Redis Session 强制**：`DEV_MODE=false` 时 `SESSION_STORAGE_BACKEND`
  必须为 `redis`，否则启动 fail-fast（不再只 warning）；Redis 初始化失败同样
  fail-fast，绝不静默回退进程内 memory。见
  `core/config.py::validate_distributed_runtime_settings`、`core/container.py`。
- **多 Worker 一致性 gate**：生产且 `GUNICORN_WORKERS>1` 时要求
  `LANGGRAPH_CHECKPOINT_BACKEND=postgres` + `SESSION_STORAGE_BACKEND=redis` +
  `AGENT_RUN_THREAD_LOCK_ENABLED=true`/`BACKEND=redis`，否则启动失败。生产还要求
  `AGENT_RUN_DISPATCH=celery`，且 `AGENT_RUN_THREAD_LOCK_TTL_SECONDS` >
  `AGENT_RUN_TASK_TIME_LIMIT` + safety margin（锁不能在任务仍在执行时过期）。
- **Reliability 语义（能力边界）**：
  - at-least-once task delivery（`task_acks_late` + `task_reject_on_worker_lost`
    + Redis `visibility_timeout`），**不是** exactly-once；
  - application-level run 幂等（终态重复投递 no-op；`idempotency_key` 唯一约束）；
  - per-thread distributed mutual exclusion（单 Redis，非 Redlock 集群）；
  - external PostgreSQL checkpoint persistence；
  - node/checkpoint-boundary durable execution（失败节点可能重新执行，节点副作用
    需幂等）；**不**宣称任意 Python 指令级无损恢复；
  - application-level **dead-letter**：`agent_dead_letters` 表（不可变历史：
    run_id / thread_id / attempt_count / error_type / error_code / entered_at）+
    `GET /api/runs/dead`；**不是** broker-native DLX。
  - **人工重放闭环**：`RunService.requeue_dead_letter` + CLI
    `python scripts/replay_dead_run.py <run_id>`。重放**复用原 run_id**（只产生新的
    队列投递）——换新 run_id 会绕过工具幂等键，把已成功的退款/改单再执行一次。
    原始 DLQ 历史不被改写，`agent_run_dead_letter_replay_total` 计数。
  - 工具侧幂等 ledger（`tool_side_effects`，`operation_key = run_id:tool_call_id`）
    + `execute_idempotent_operation` 包裹执行；`ToolRegistry.register(side_effect=True)`
    的**写工具**在 Run 执行上下文内自动走 ledger（`tools/tool_registry.py`），
    `agent/base_agent.py` 已把 LLM 的 `tool_call_id` 透传到 registry；
    PENDING 认领租约未过期时抛 `TransientError`（退避重投）而不是重复执行副作用。
    外部系统端到端幂等仍需下游 API 接受 idempotency key。
  - **断点续跑语义（已修）**：LangGraph `ainvoke(state, cfg)` 会**从 START 重新
    执行**并覆盖 channel 值；只有 `ainvoke(None, cfg)` 才从 checkpoint 的 `next`
    续跑。`runtime/bootstrap.py::invoke_graph_with_resume` 统一判定：有未完成
    checkpoint（`next` 非空）则续跑，否则正常执行；API 快路径
    （`api/app.py::_pending_steps`）同语义。此前 worker 崩溃后是"从头重跑"。
  - thread lease **执行期间续租**（`runtime/executor.py::_heartbeat_loop` 同时续 DB
    ownership lease 与 Redis lock TTL，`agent_thread_lease_renewed_total` /
    `agent_worker_heartbeat` 观测）。
  - **worker-owned 状态迁移的 owner CAS**：worker 提交 AgentRun 状态
    （`mark_succeeded` / `mark_failed` / `mark_retrying` / `mark_waiting_approval` /
    RUNNING 来源的 `mark_dead_letter`）走 `repository.transition_owned()`，
    `run_id + status + worker_id + lease 未过期` 在**同一条 UPDATE** 里判定；
    `heartbeat()` 是原子 owner CAS（`renew_lease_owned`），RUNNING 接管是原子
    谓词（`takeover_running`），两个竞争者只有一个能接管。因此 **ownership 丢失后
    旧 worker 无法再提交状态**，也**不能靠迟到的续租给自己续命**（严格 lease
    expiry）。失去所有权抛 `RunOwnershipLost`，executor 读到当前状态即退出：
    不记失败、不重试、不进 DLQ、不消耗 attempt。外部路径（`cancel_run()`、
    DLQ 重放 / reconciler / 管理 API）**不**要求匹配 worker。
    证据：`tests/unit/test_agent_run_runtime.py` + 真实 PostgreSQL 下的
    `tests/integration/runtime/test_worker_ownership_cas.py`。
  - 仍**没有** fencing token，且**没有**强制中止：pause 超过 TTL 的旧 worker
    **不会被强制 abort**，其协程可能继续跑完；它的**外部节点副作用**也不受
    owner CAS 保护（仍依赖 side-effect ledger 的幂等）。也就是说本层解决的是
    「stale worker 不能提交 AgentRun 状态」，**不是**完整的 stale-worker fencing，
    更不是 exactly-once（投递仍是 at-least-once）。
  - **Run 事件流（新增）**：worker 写 Redis Stream `agent:run:{run_id}:events`
    （`runtime/events.py`），API 通过 `GET /api/runs/{run_id}/events` 以 SSE 转发，
    支持 `Last-Event-ID` 断点续读。事件负载走**字段白名单**，query/用户标识/凭据
    不入事件流。定位是**观测通道而非业务真相源**：best-effort resumable，
    **不是** exactly-once——单连接内不重复，但用较旧 `Last-Event-ID` 重连会重放
    已处理事件（重复），`replay=false` 与 idle 超时会造成缺口，`MAXLEN` 近似裁剪
    后的历史不可恢复（且近似裁剪不是硬上界）。
- **SSE + checkpoint 序列化修复（P0）**：`stream_callback` 曾作为 LangGraph channel
  （`core/state.py`）随 state 一起被 checkpointer 序列化，因是 async 可调用对象而
  抛 `TypeError: Type is not msgpack serializable: function` —— MemorySaver 与官方
  `AsyncPostgresSaver` **都**失败，生产默认 postgres 后端下 `/api/chat/stream`
  必然报错。现改为 contextvar 传递（`core/streaming_context.py`），节点经
  `get_stream_callback(state)` 读取（仍兼容从 state 取值的老调用方）。
  CI VERIFIED：真实 PG checkpointer 下 SSE 正常完成、`done` 帧、0 错误帧；
  回归测试 `tests/integration/runtime/test_sse_checkpoint_serialization.py`。
- **能力层级**：
  - Level 1（已实现，代码存在）：Postgres checkpoint、Redis session、Redis
    per-thread lock、AgentRun 真相源、Celery + Redis broker、worker execution、
    run_id dispatch、acks_late、reject_on_worker_lost、visibility_timeout、retry
    基础、tool ledger、idempotency helper、Prometheus metrics。
  - Level 2（本地验证，有命令 + artifact，真实 PG + Redis + 真实多进程 Celery）：
    `make runtime-e2e`（tests/integration/runtime，覆盖 checkpoint
    跨进程恢复 / thread-run 分离 / 同 thread 执行区间不重叠 / 跨 thread 并发耗时 /
    lease 非 owner 释放与 TTL 接管 / queue-worker 解耦 / worker kill -9 后从
    checkpoint 续跑 / 三次重试后成功 / permanent 不重试 / DLQ + 重放 / 副作用工具
    重复投递下的副作用去重 / 事件流与 SSE 续读）与 `make runtime-chaos`
    （结构化证据 JSON）。
    CI：`.github/workflows/ci.yml` 的 `runtime-e2e` job（postgres + redis service
    container）。两者的证据等级是 **CI VERIFIED**，生产集群仍 NOT_VERIFIED。
  - Level 3（未生产验证）：真实生产集群 / 多副本长期稳定 / 真实用户流量 /
    真实 ERP 写操作 / 大规模 queue backlog / K8s autoscaling / multi-region。
- **Evidence SHA 语义**：artifact schema `distributed-runtime-evidence/v2` 使用
  `tested_code_sha`（生成 evidence 时的被测代码 commit）+ `generated_at` +
  `overall_status`；artifact 自身随后提交到另一个 commit（生成时无法预知，
  `artifact_commit_sha` 为 null）。历史 v1 artifact 保留不改。
- **状态归属**：见 [runtime-state-ownership](../design/runtime-state-ownership.md)
  与 [ADR-009](../decisions/009-distributed-agent-runtime.md)。仍未实现的只有
  worker pool autoscaling / backpressure admission control / Kubernetes-HPA /
  multi-region；**DLQ 运维闭环已实现**（`GET /api/runs/dead` +
  `RunService.requeue_dead_letter` + `scripts/replay_dead_run.py` 复用原 `run_id`），
  **Run 事件流 SSE bridge 已实现**（`runtime/events.py` Redis Stream +
  `GET /api/runs/{run_id}/events`，支持 `Last-Event-ID` 续读，但只是
  best-effort 观测通道，不是业务真相源）。
  深化设计见
  [async-agent-worker-architecture](../design/async-agent-worker-architecture.md)。
- **验证命令**（需要真实 Redis/PostgreSQL；默认 skip）：
  ```bash
  TEST_REDIS_URL=redis://localhost:6379 \
  TEST_DISTRIBUTED_DB_URL=postgresql://postgres:postgres@localhost:5432/cosmetics_ai \
      pytest tests/integration/test_worker_crash_recovery.py tests/integration/test_thread_lock_redis.py -q
  # 生成机器可读 evidence（artifacts/distributed-runtime/<ts>/report.json）
  TEST_DISTRIBUTED_DB_URL=postgresql://postgres:postgres@localhost:5432/cosmetics_ai \
  TEST_REDIS_URL=redis://localhost:6379 \
      python scripts/verify_distributed_runtime.py
  # 或脚本
  scripts/repro_worker_crash_recovery.sh
  ```
  设计细节：[docs/design/distributed-agent-runtime.md](../design/distributed-agent-runtime.md)。

## 配置层级语义（runtime fallback ≠ 模板推荐值）

- `runtime fallback`: `core/config.py` 中 `os.getenv(...)` 的默认值。
- `deployment recommended value`: `.env.example` 与 `deploy/compose/` 模板值。
  两者可以不同，模板覆盖处必须注释说明（例如 `LLM_MAX_TOKENS`、`HTTP_TIMEOUT`、
  `LLM_ROUTER_TIMEOUT`）。
- `test override` / `production override`: 各自的环境配置文件显式覆盖。
- 任何一侧漂移会被 `scripts/audit_doc_consistency.py` 的 canonical config 检查捕获。

## 证据边界

- 缓存命中率（L1/L2/L3 总）与缓存各层访问延迟：观测端点/指标存在
  （`/api/cache/stats`、`cache_redis_latency_seconds` / `cache_qdrant_latency_seconds` /
  `cache_operation_duration_seconds`），但生产级数值均 `NOT_MEASURED`；
  缓存命中的收益语义是"跳过 Router/Agent/LLM 链路"，不宣称具体延迟数字。
- Provider authentication、provider token usage/billing、生产延迟/SLA、FCR、
  人工效率：`NOT_VERIFIED` / `NOT_MEASURED`（除非链接带 provenance 的当前 artifact）。
- 本地测试/fixture benchmark ≠ 生产证据。详见
  [docs/evaluation/production-evidence.md](../evaluation/production-evidence.md)。
- 历史报告快照位于 `docs/reports/**`（含日期），不作为当前事实入口。
- **RAG evidence 状态（当前 649-query 正式指标）**：当前 `NOT_VERIFIED`。
  该状态由 `scripts/rag_evidence_status.py` 从
  `artifacts/evaluation/rag-649/**/report.json` 中满足 formal full-run
  contract 的 artifact 动态推导（preflight-only / smoke subset artifact 与
  文档声明都不构成正式指标证据），文档行的状态必须与推导一致；
  已提交的 preflight artifact（provider auth blocker）时间点有效，
  不构成正式指标。
  RAG 评估的具体生命周期由 `scripts/project_facts.py`（benchmark 数/harness 事实）
  与 [docs/reference/rag-evaluation.md](rag-evaluation.md) 共同维护。

## 历史快照与当前事实的关系

- `docs/reports/plans/**`、`docs/reports/milestone/**`、`docs/reports/releases/**`
  均为 AUDIT / RELEASE SNAPSHOT，只在该快照执行时有效。
- Snapshot SHA != Current HEAD（用 `git rev-parse HEAD` 获取当前 HEAD）。
- ADR 的历史决策正文不修改；当决策被取代时只更新 Status 行（例如 ADR-003 被
  [ADR-007](../decisions/007-current-default-llm.md) 取代）。
