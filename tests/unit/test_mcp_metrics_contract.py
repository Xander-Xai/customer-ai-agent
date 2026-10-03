"""MCP Prometheus 指标契约（真实 ``prometheus_client``，进程内断言）。

锁三件事：

1. **声明与观测的 label 名逐字一致**。
   ``core/monitoring.py`` 声明 ``labelnames``、``tools/mcp_adapter.py::_inc/_observe``
   传 ``**labels``。两者不一致时，真实 client 的 ``labels()`` 抛
   ``ValueError: No label names were set when constructing histogram:...``；
   该异常被指标降级路径 ``except Exception: pass`` **静默吞掉**，指标既不报错也不出样本
   —— 直方图看起来"在"，实际永远恒为 0。这正是 PR #42 review 抓到的问题
   （``mcp_tool_duration_seconds`` 声明无 label，观测却带 ``server`` / ``tool``）。
   所以这里必须用**真实** client 断言样本真的产生：把 ``.labels()`` mock 成永远返回
   self 的替身（``_NoopMetric``）永远发现不了这个问题。
2. **失败路径也留样本**。``invoke`` 的耗时观测在 ``finally`` 里，timeout / error /
   tool_error 都必须计入延迟分布，否则超时这类最该被观测的慢调用恰好不可见。
3. **不引入高基数 label**，且远端提供的字符串**绝不**成为 label 值
   （不在 allowlist 的工具名不进 ``tool`` label，否则 server 暴露新工具就能无界
   扩张时序数）。

只覆盖 MCP 家族指标，不动其它 metric。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY, Counter, Gauge, Histogram  # noqa: E402

from core import monitoring  # noqa: E402
from core.hitl.risk import RiskLevel  # noqa: E402
from tools.mcp_adapter import (  # noqa: E402
    MCPServerConfig,
    MCPTimeoutError,
    MCPToolAdapter,
    MCPToolExecutionError,
    MCPUnauthorizedToolError,
)

MCP_ADAPTER_PATH = Path(__file__).resolve().parents[2] / "tools" / "mcp_adapter.py"

#: 本文件专属的 ``server`` label 值。
#:
#: ``REGISTRY`` 是**进程级全局**的，而本文件用绝对计数（``== 1`` / ``== 3``）证明
#: 样本真的产生——这些断言只在对应 label 组合**只被本文件写过**时成立。
#: 其它 MCP 测试文件（如 ``test_mcp_adapter_runtime.py``）也用 ``server="catalog"`` /
#: ``tool="get_sku"`` 观测同一个直方图，两者落进同一进程就会互相污染计数，且结果
#: 取决于 pytest 的文件执行顺序。给本文件一个专属 server 名即可互不干扰，与
#: ``TestCardinalityDiscipline`` 里 ``catalog_cardinality`` 的用法同一理由。
_METRICS_CONTRACT_SERVER = "metrics_contract_catalog"

# 声明侧的权威契约。任何一个 MCP 指标的 label 名变化都必须同步改这里，且必须先
# 回答"这个值是否由配置决定、是否有界"。
EXPECTED_MCP_LABELS: dict[str, tuple[str, ...]] = {
    "mcp_tool_call_total": ("server", "tool", "status"),
    "mcp_tool_error_total": ("server", "tool", "reason"),
    "mcp_tool_duration_seconds": ("server", "tool"),
    "mcp_tool_register_total": ("server", "outcome"),
}

# 每请求唯一 / 远端不可信的值一律不得成为 label 名（与
# tests/unit/test_runtime_metrics_contract.py 的 FORBIDDEN_LABELS 同一纪律）。
FORBIDDEN_LABEL_NAMES = {
    "run_id",
    "thread_id",
    "user_id",
    "query",
    "customer_query",
    "session_id",
    "idempotency_key",
    "tool_call_id",
    "error_message",
    "arguments",
    "result",
}


class _FakeMCPClient:
    """最小 ``MCPClientProtocol`` 实现：返回固定 SDK 形态结果，或抛指定异常。"""

    def __init__(self, *, result=None, error: BaseException | None = None):
        self._result = result
        self._error = error

    async def list_tools(self):
        return []

    async def call_tool(self, name: str, arguments: dict):
        if self._error is not None:
            raise self._error
        return self._result

    async def close(self):
        return None


def _ok_result(text: str = "sku-1"):
    """``CallToolResult`` 的 snake_case 形态（SDK 2.x）。"""
    return {"is_error": False, "content": [{"text": text}]}


def _adapter(server: str, tools: tuple[str, ...], client, **overrides) -> MCPToolAdapter:
    return MCPToolAdapter(
        MCPServerConfig(name=server, allowed_tools=tools, risk_level=RiskLevel.LOW, **overrides),
        client=client,
    )


def _labelnames(metric_name: str) -> tuple[str, ...]:
    return tuple(getattr(monitoring, metric_name)._labelnames)


def _duration_count(**labels: str) -> float | None:
    return REGISTRY.get_sample_value("mcp_tool_duration_seconds_count", labels or None)


def _observed_label_names() -> dict[str, set[str]]:
    """AST 扫描 ``tools/mcp_adapter.py``，取每个 ``_inc`` / ``_observe`` 调用点
    实际传入的 label 名。

    用 AST 而非字符串搜索：Python 里 kwarg 名是字面量，取值（``tool=mcp_name``）
    不影响要断言的 label 名，所以扫描精确，且不需要执行被测代码。
    """
    tree = ast.parse(MCP_ADAPTER_PATH.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in ("_inc", "_observe"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        metric_name = node.args[0].value
        assert isinstance(metric_name, str), f"指标名必须是字面量: {metric_name!r}"
        if not metric_name.startswith("mcp_"):
            continue
        found.setdefault(metric_name, set()).update(kw.arg for kw in node.keywords if kw.arg)
    return found


class TestDeclarationMatchesObservation:
    """声明侧 label 名 == 调用侧 label 名（逐字、逐个）。"""

    def test_observation_call_sites_cover_every_declared_metric(self):
        # 反向也锁：新增 MCP 指标却忘了登记契约，会让下面的逐指标比对静默通过。
        assert set(_observed_label_names()) == set(EXPECTED_MCP_LABELS)

    @pytest.mark.parametrize("metric_name", sorted(EXPECTED_MCP_LABELS))
    def test_declared_labelnames_equal_labels_passed_at_call_sites(self, metric_name):
        observed = _observed_label_names()[metric_name]
        declared = set(_labelnames(metric_name))
        assert declared == observed, (
            f"{metric_name} 声明 label {sorted(declared)} != 观测 label {sorted(observed)}；"
            "真实 prometheus_client 会抛 ValueError 且被 except 吞掉 -> 静默零样本"
        )

    @pytest.mark.parametrize("metric_name", sorted(EXPECTED_MCP_LABELS))
    def test_metrics_are_real_client_objects(self, metric_name):
        """降级替身（prometheus_client 缺失时的 ``_NoopMetric``）下本文件其余断言都
        恒真，所以先确认真实 client 在场。"""
        metric = getattr(monitoring, metric_name)
        expected_type = Histogram if metric_name.endswith("_seconds") else Counter
        assert isinstance(metric, expected_type)

    @pytest.mark.parametrize("metric_name", sorted(EXPECTED_MCP_LABELS))
    def test_no_forbidden_high_cardinality_labels(self, metric_name):
        assert not (set(_labelnames(metric_name)) & FORBIDDEN_LABEL_NAMES)

    def test_duration_histogram_is_a_real_client_metric_with_contract_labels(self):
        metric = monitoring.mcp_tool_duration_seconds
        assert isinstance(metric, Histogram)
        assert _labelnames("mcp_tool_duration_seconds") == ("server", "tool")


class TestDurationSamplesActuallyRecorded:
    """真实 client + 真实 ``invoke``：断言样本真的产生（PR #42 缺的正是这条证据）。"""

    async def test_successful_invoke_records_labelled_duration_sample(self):
        server, tool = _METRICS_CONTRACT_SERVER, "get_sku"
        adapter = _adapter(server, (tool,), _FakeMCPClient(result=_ok_result()))

        assert await adapter.invoke(tool, {"sku": "X1"}) == "sku-1"
        assert _duration_count(server=server, tool=tool) == 1

    async def test_timeout_still_records_duration_sample(self):
        """耗时观测在 ``finally`` 里：超时是最该被看到的慢调用，绝不能因为抛异常
        而从延迟分布里消失。"""
        server, tool = _METRICS_CONTRACT_SERVER, "get_sku_slow"
        adapter = _adapter(
            server,
            (tool,),
            _FakeMCPClient(error=TimeoutError()),
            timeout_seconds=0.01,
        )

        with pytest.raises(MCPTimeoutError):
            await adapter.invoke(tool, {})
        assert _duration_count(server=server, tool=tool) == 1

    async def test_tool_error_result_still_records_duration_sample(self):
        server, tool = _METRICS_CONTRACT_SERVER, "get_sku_toolerr"
        adapter = _adapter(
            server,
            (tool,),
            _FakeMCPClient(result={"is_error": True, "content": [{"text": "boom"}]}),
        )

        with pytest.raises(MCPToolExecutionError):
            await adapter.invoke(tool, {})
        assert _duration_count(server=server, tool=tool) == 1

    async def test_histogram_count_sum_and_inf_bucket_advance_together(self):
        """只让 count 涨、sum 不动（或 +Inf 桶不动）说明 observe 没落到真实 client。"""
        server, tool = _METRICS_CONTRACT_SERVER, "get_sku_stats"
        adapter = _adapter(server, (tool,), _FakeMCPClient(result=_ok_result()))
        for _ in range(3):
            await adapter.invoke(tool, {})

        labels = {"server": server, "tool": tool}
        assert _duration_count(**labels) == 3
        assert REGISTRY.get_sample_value("mcp_tool_duration_seconds_sum", labels) > 0.0
        assert (
            REGISTRY.get_sample_value("mcp_tool_duration_seconds_bucket", {**labels, "le": "+Inf"})
            == 3
        )

    async def test_duration_series_are_distinct_per_server_and_tool(self):
        """label 生效的另一半：不同 server/tool 必须是不同时间序列，而不是被聚合成
        一条无 label 序列（无 label 正是"声明漏了 label"的退化形态）。"""
        adapter_a = _adapter("srva", ("get_sku",), _FakeMCPClient(result=_ok_result()))
        adapter_b = _adapter("srvb", ("get_sku",), _FakeMCPClient(result=_ok_result()))
        await adapter_a.invoke("get_sku", {})
        await adapter_b.invoke("get_sku", {})

        assert _duration_count(server="srva", tool="get_sku") == 1
        assert _duration_count(server="srvb", tool="get_sku") == 1

    async def test_counters_sharing_the_same_labels_also_land(self):
        """同一个 ``invoke`` 里的 counter 有声明 label，行为应当一致；
        顺带证明 duration 的失败不是普遍性的 client 问题。"""
        server, tool = _METRICS_CONTRACT_SERVER, "get_sku_counter"
        adapter = _adapter(server, (tool,), _FakeMCPClient(result=_ok_result()))
        await adapter.invoke(tool, {})

        assert (
            REGISTRY.get_sample_value(
                "mcp_tool_call_total", {"server": server, "tool": tool, "status": "ok"}
            )
            == 1
        )


class TestCardinalityDiscipline:
    """远端提供的字符串不得成为 label 值；不新增维度。"""

    async def test_rejected_remote_tool_name_never_becomes_a_label(self):
        # 用专属 server 名：REGISTRY 是进程级全局的，label 值撞车会读到别的用例的
        # 计数。
        server = "catalog_cardinality"
        adapter = _adapter(server, ("get_sku",), _FakeMCPClient(result=_ok_result()))

        with pytest.raises(MCPUnauthorizedToolError):
            await adapter.invoke("remote_only_tool", {})

        assert _duration_count(server=server, tool="remote_only_tool") is None
        # allowlist 内的工具走完后该 label 才允许出现。
        await adapter.invoke("get_sku", {})
        assert _duration_count(server=server, tool="get_sku") == 1

    def test_duration_labels_stay_within_config_bounded_dimensions(self):
        assert set(_labelnames("mcp_tool_duration_seconds")) <= {"server", "tool"}

    def test_fix_does_not_add_new_metric_families(self):
        """只改 declaration：不新增 / 不重命名 metric family，避免同名不同 label 的
        双 family 冲突与无意的时序新增。

        注意 Prometheus 的 family / 样本命名差异：``Counter("x_total")`` 的 family 名
        是 ``x``，而 ``Histogram("x_seconds")`` 的 family 名就是 ``x_seconds``。
        """
        families = {m.name for m in REGISTRY.collect() if m.name.startswith("mcp_")}
        assert families <= {
            "mcp_tool_call",
            "mcp_tool_error",
            "mcp_tool_duration_seconds",
            "mcp_tool_register",
        }


def test_unrelated_gauge_untouched_by_this_fix():
    """本次修复不涉及 gauge；留一条断言防止改动扩散到其它指标。"""
    assert isinstance(monitoring.agent_approval_pending, Gauge)
    assert monitoring.agent_approval_pending._labelnames == ()
