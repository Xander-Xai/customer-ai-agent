"""MCP 初始化的事务边界：**fail-closed 失败时必须整体回滚**（回归 PR #42 review）。

背景：``ServiceContainer._init_mcp_tools`` 逐个 server 连接并把工具注册进
**共享的** ``ToolRegistry``。当配置多个 server 时，A 成功、B 失败会让
「本次 attempt 已建立的部分状态」与「容器应当处于的状态」分叉。修复前：

- ``raise ConfigurationError`` 只关掉**失败的那个** adapter，
- A 已经拉起的 stdio 子进程 / SSE 会话留在原地（孤儿进程），
- A 已注册进 ``tool_registry`` 的工具**留在 registry 里**（指向已死的 adapter，
  LLM 仍会看到并可被调用 → ``MCPUnavailableError``），
- ``self.mcp_adapters`` 非空成为「已初始化」的假信号，
- 且 ``close()`` 因 ``_initialized`` 仍为 ``False`` 会**提前 return**，
  连 shutdown 兜底都不存在 → 泄漏是进程生命周期内的**永久**泄漏。

本文件钉死的是**回滚的原子性**，不是 MCP 传输行为，因此全部用注入式
fake client（不 spawn 子进程、不依赖 ``mcp`` SDK）。
"""

from __future__ import annotations

import json

import pytest

from core.container import ServiceContainer
from core.hitl.risk import RiskLevel
from tools.mcp_adapter import (
    MCPServerConfig,
    MCPToolAdapter,
    MCPUnavailableError,
)
from tools.tool_registry import SOURCE_NATIVE, ToolRegistry

pytestmark = pytest.mark.unit


def _configuration_error() -> type[Exception]:
    """在**运行时**解析 ``core.config.ConfigurationError`` 的当前类对象。

    刻意不在模块作用域 import：``tests/unit/test_hitl_api.py`` 会对
    ``core.config`` 做 ``importlib.reload``，而 reload 会**重建**模块里的类对象。
    模块作用域绑定的旧类与容器在调用时 ``from core.config import
    ConfigurationError`` 拿到的类不是同一个类型，``pytest.raises`` 会静默不匹配 ——
    表现为「整个 tests/unit 跑一遍就红，单跑就绿」。

    这也是本文件所有 MCP 旋钮都用 ``monkeypatch.setattr`` 字符串目标、
    在调用时解析的原因：``core.config`` 是可被整体 reload 的模块。
    """
    import core.config

    return core.config.ConfigurationError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeToolDescriptor:
    """MCP ``tools/list`` 返回项的最小替身（字段名走 snake_case 别名）。"""

    def __init__(self, name: str, description: str = "d"):
        self.name = name
        self.description = description
        self.input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}


class FakeClient:
    """注入式 MCP client：记录连接/关闭，供断言生命周期用。

    ``fail_on_list`` 模拟「进程拉起了、握手成功，但 ``tools/list`` 失败」——
    这是最容易泄漏子进程的失败点：此时 adapter 侧已经持有真实 session。
    """

    def __init__(self, tools=None, *, fail_on_list: bool = False):
        self._tools = tools if tools is not None else [FakeToolDescriptor("search")]
        self.fail_on_list = fail_on_list
        self.connected = False
        self.close_calls = 0

    async def connect(self):
        self.connected = True

    async def list_tools(self):
        if self.fail_on_list:
            # ConnectionError → 被 adapter 的 _map_error 归类为 MCPUnavailableError
            raise ConnectionError("server B 子进程握手后 tools/list 失败")
        return self._tools

    async def call_tool(self, name, arguments):
        return {"ok": name}

    async def close(self):
        self.close_calls += 1
        self.connected = False


class TrackingAdapter(MCPToolAdapter):
    """记录 ``close()`` 调用次数的 adapter。

    为什么在 **adapter** 层而不是 client 层断言：``MCPToolAdapter.close()`` 只关闭
    **自己创建** 的 client（注入的 fake 由注入方代管，见 ``_owns_client``）。而生产里
    真正会泄漏的是 ``build_mcp_adapters`` 造出的、自带 SDK client / stdio 子进程的
    adapter。容器这一层应尽的义务只有一个：**对本次 attempt 的每个 adapter 调用
    ``close()``** —— 所有权与子进程回收是 adapter 自己的职责。所以断言边界在 adapter。
    """

    def __init__(self, config, client=None, *, close_error: Exception | None = None):
        super().__init__(config, client=client)
        self.close_calls = 0
        self._close_error = close_error

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error
        await super().close()


def _config(name: str) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command="unused-in-these-tests",
        allowed_tools=["search"],
        risk_level=RiskLevel.LOW,
    )


def _make_registry_with_native_tool() -> ToolRegistry:
    registry = ToolRegistry()

    async def _native_handler(arguments: dict) -> str:
        return "native"

    registry.register(
        name="query_product",
        description="native tool",
        parameters={"type": "object", "properties": {}},
        handler=_native_handler,
        source=SOURCE_NATIVE,
    )
    return registry


def _servers_json(*names: str) -> str:
    return json.dumps(
        [
            {
                "name": n,
                "transport": "stdio",
                "command": "unused-in-these-tests",
                "allowed_tools": ["search"],
                "risk_level": "low",
            }
            for n in names
        ]
    )


def _install(container: ServiceContainer, adapters, monkeypatch) -> None:
    """把 MCP 旋钮 + adapter 列表接到容器上（不碰真实 MCP 传输）。"""
    import core.config as config_mod
    import tools.mcp_adapter as adapter_mod

    monkeypatch.setattr(config_mod, "MCP_ENABLED", True, raising=False)
    monkeypatch.setattr(config_mod, "MCP_FAIL_CLOSED", True, raising=False)
    monkeypatch.setattr(adapter_mod, "build_mcp_adapters", lambda configs: list(adapters))


@pytest.fixture
def container() -> ServiceContainer:
    c = ServiceContainer()
    c.tool_registry = _make_registry_with_native_tool()
    return c


# ---------------------------------------------------------------------------
# 1. 最小复现：A 成功 + B 失败 + fail_closed
# ---------------------------------------------------------------------------


class TestFailClosedRollsBackWholeAttempt:
    @pytest.mark.asyncio
    async def test_prior_server_adapter_is_closed(self, container, monkeypatch):
        """A 的会话 / 子进程必须被关掉，不能留在原地变成孤儿进程。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()):
            await container._init_mcp_tools()

        assert adapter_a.close_calls == 1, "失败前已连接的 server A 的 session 未被释放"
        assert adapter_b.close_calls == 1, "失败的 server B 自身的会话未被释放"

    @pytest.mark.asyncio
    async def test_prior_server_tools_are_removed(self, container, monkeypatch):
        """A 已注册的 MCP 工具必须撤销——否则 LLM 能调到已死的 adapter。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()):
            await container._init_mcp_tools()

        assert "mcp__alpha__search" not in container.tool_registry.list_tools()
        assert container.tool_registry.list_tools() == ["query_product"]

    @pytest.mark.asyncio
    async def test_native_tools_are_never_removed(self, container, monkeypatch):
        """回滚只能撤销**本次新增**的 MCP 工具，绝不误伤 native 工具。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()):
            await container._init_mcp_tools()

        assert "query_product" in container.tool_registry.list_tools()
        assert container.tool_registry.source_for("query_product") == SOURCE_NATIVE

    @pytest.mark.asyncio
    async def test_container_keeps_no_partial_mcp_state(self, container, monkeypatch):
        """容器不得进入 partial MCP state：``mcp_adapters`` 必须为空。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()):
            await container._init_mcp_tools()

        assert container.mcp_adapters == []

    @pytest.mark.asyncio
    async def test_rollback_survives_a_raising_close(self, container, monkeypatch):
        """A 的 close() 自己炸掉时，B 的失败仍要照常抛出（不被 close 异常顶掉）。"""
        adapter_a = TrackingAdapter(
            _config("alpha"), client=FakeClient(), close_error=RuntimeError("close boom")
        )
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()) as exc:
            await container._init_mcp_tools()

        assert "beta" in str(exc.value)
        assert container.mcp_adapters == []
        # A 的 close 失败不能中断清理：后面的 B 仍然要关。
        assert adapter_b.close_calls == 1

    @pytest.mark.asyncio
    async def test_original_cause_is_preserved(self, container, monkeypatch):
        """保留原始 failure semantics：异常不吞，且整条 ``__cause__`` 链指向真实根因。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()) as exc:
            await container._init_mcp_tools()

        assert "beta" in str(exc.value)
        cause = exc.value.__cause__
        assert isinstance(cause, MCPUnavailableError)
        assert isinstance(cause.__cause__, ConnectionError)


# ---------------------------------------------------------------------------
# 2. Retry：残留的 mcp_adapters 不得让重试跳过初始化
# ---------------------------------------------------------------------------


class TestRetryAfterFailedInit:
    @pytest.mark.asyncio
    async def test_retry_reinitializes_instead_of_skipping(self, container, monkeypatch):
        """第一次失败回滚干净后，重试必须**真正重新初始化**，而不是被
        「``mcp_adapters`` 非空 → 已初始化过」短路掉。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        with pytest.raises(_configuration_error()):
            await container._init_mcp_tools()

        # 第二次：运维修好了 beta，两个 server 都健康。
        retry_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        retry_b = TrackingAdapter(_config("beta"), client=FakeClient())
        monkeypatch.setattr(
            "tools.mcp_adapter.build_mcp_adapters",
            lambda configs: [retry_a, retry_b],
        )

        await container._init_mcp_tools()

        assert "mcp__alpha__search" in container.tool_registry.list_tools()
        assert "mcp__beta__search" in container.tool_registry.list_tools()
        assert container.mcp_adapters == [retry_a, retry_b]
        assert retry_a.close_calls == 0

    @pytest.mark.asyncio
    async def test_successful_init_is_still_idempotent(self, container, monkeypatch):
        """成功路径的既有幂等语义不得被回滚逻辑破坏。"""
        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient())

        monkeypatch.setattr("core.config.MCP_SERVERS", _servers_json("alpha", "beta"))
        _install(container, [adapter_a, adapter_b], monkeypatch)

        await container._init_mcp_tools()
        tools_after_first = container.tool_registry.list_tools()

        # 换一个 adapter 集合再调一次：不应重复连接 / 重复注册。
        await container._init_mcp_tools()

        assert container.tool_registry.list_tools() == tools_after_first
        assert len(container.mcp_adapters) == 2


# ---------------------------------------------------------------------------
# 3. 非 fail-closed 降级路径不受影响
# ---------------------------------------------------------------------------


class TestFailOpenDegradationUnchanged:
    @pytest.mark.asyncio
    async def test_fail_open_keeps_healthy_server_and_drops_broken_one(
        self, container, monkeypatch
    ):
        """``MCP_FAIL_CLOSED=false``：A 保留，B 关闭后跳过——行为不变。"""
        import core.config as config_mod

        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))

        monkeypatch.setattr(config_mod, "MCP_ENABLED", True, raising=False)
        monkeypatch.setattr(config_mod, "MCP_FAIL_CLOSED", False, raising=False)
        monkeypatch.setattr(config_mod, "MCP_SERVERS", _servers_json("alpha", "beta"))
        monkeypatch.setattr(
            "tools.mcp_adapter.build_mcp_adapters",
            lambda configs: [adapter_a, adapter_b],
        )

        await container._init_mcp_tools()

        assert container.mcp_adapters == [adapter_a]
        assert "mcp__alpha__search" in container.tool_registry.list_tools()
        assert "mcp__beta__search" not in container.tool_registry.list_tools()
        # 降级路径只关失败者，不动已成功的 A。
        assert adapter_b.close_calls == 1
        assert adapter_a.close_calls == 0

    @pytest.mark.asyncio
    async def test_fail_open_later_failure_does_not_drop_earlier_tools(
        self, container, monkeypatch
    ):
        """A、B、C，B 失败：fail-open 下 A 的工具必须留下（不是回滚语义）。"""
        import core.config as config_mod

        adapter_a = TrackingAdapter(_config("alpha"), client=FakeClient())
        adapter_b = TrackingAdapter(_config("beta"), client=FakeClient(fail_on_list=True))
        adapter_c = TrackingAdapter(_config("gamma"), client=FakeClient())

        monkeypatch.setattr(config_mod, "MCP_ENABLED", True, raising=False)
        monkeypatch.setattr(config_mod, "MCP_FAIL_CLOSED", False, raising=False)
        monkeypatch.setattr(config_mod, "MCP_SERVERS", _servers_json("alpha", "beta", "gamma"))
        monkeypatch.setattr(
            "tools.mcp_adapter.build_mcp_adapters",
            lambda configs: [adapter_a, adapter_b, adapter_c],
        )

        await container._init_mcp_tools()

        assert container.mcp_adapters == [adapter_a, adapter_c]
        tools = container.tool_registry.list_tools()
        assert "mcp__alpha__search" in tools
        assert "mcp__gamma__search" in tools
        assert "mcp__beta__search" not in tools


# ---------------------------------------------------------------------------
# 4. ToolRegistry.unregister 契约
# ---------------------------------------------------------------------------


class TestRegistryUnregister:
    def test_unregister_removes_tool(self):
        registry = ToolRegistry()

        async def _h(arguments: dict) -> str:
            return "x"

        registry.register("t", "d", {"type": "object", "properties": {}}, _h)
        assert "t" in registry.list_tools()

        registry.unregister("t")

        assert "t" not in registry.list_tools()

    def test_unregister_unknown_tool_is_noop(self):
        registry = ToolRegistry()
        registry.unregister("never-registered")
        assert registry.list_tools() == []

    def test_execute_reports_missing_tool_after_unregister(self):
        registry = ToolRegistry()

        async def _h(arguments: dict) -> str:
            return "x"

        registry.register("t", "d", {"type": "object", "properties": {}}, _h)
        registry.unregister("t")

        out = __import__("asyncio").run(registry.execute("t", {}))
        assert "不存在" in out
