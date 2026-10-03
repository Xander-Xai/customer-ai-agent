"""``McpSdkClient`` 契约（PR #42 覆盖率修复）。

为什么用 stub 而不是真装 ``mcp``：

- ``mcp`` 在 ``requirements-optional.txt``（``MCP_ENABLED`` 默认 false，核心功能
  不依赖它）。CI 的 ``test`` job 只装 ``requirements.txt``，因此
  ``McpSdkClient`` 的 ~100 条语句在那条 lane 上**结构性地一行都跑不到**——
  这正是 PR #42 把 coverage 拖到 79.84% 的原因之一。
- 这里要验的是**本仓自己**的逻辑，不是第三方 SDK 的行为：SDK 版本双读
  （1.x camelCase / 2.x snake_case）、read timeout 余量、握手超时换算、
  懒加载缺失时的降级、close 的逆序与幂等、错误分类。用 stub 恰好把这些逻辑
  单独暴露出来，且不把 ``mcp`` 变成测试的硬依赖。

stub 通过 ``sys.modules`` 注入，作用域限于每个测试；``mcp`` 真装了也不会影响
这些断言（stub 是唯一被解析到的对象）。
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from core.hitl.risk import RiskLevel
from tools.mcp_adapter import (
    MCPConfigurationError,
    McpSdkClient,
    MCPServerConfig,
    MCPTimeoutError,
    MCPUnavailableError,
)

# ---------------------------------------------------------------------------
# stub SDK
# ---------------------------------------------------------------------------


class _CM:
    """记录进入/退出顺序的 async context manager。"""

    def __init__(self, label: str, log: list[str], *, enter_error=None, exit_error=None):
        self.label = label
        self.log = log
        self.enter_error = enter_error
        self.exit_error = exit_error

    async def __aenter__(self):
        self.log.append(f"enter:{self.label}")
        if self.enter_error is not None:
            raise self.enter_error
        return (f"read-{self.label}", f"write-{self.label}")

    async def __aexit__(self, *_exc):
        self.log.append(f"exit:{self.label}")
        if self.exit_error is not None:
            raise self.exit_error


class _StdioServerParameters:
    """与官方 SDK 同签名的参数对象（真实实现是 pydantic model）。"""

    def __init__(self, command, args):
        self.command = command
        self.args = args


def _install_stub_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session: Any = None,
    stdio_cm=None,
    sse_cm=None,
    log: list[str] | None = None,
    session_kwargs: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """把一个最小 ``mcp`` 包注入 ``sys.modules``，返回 (顺序日志, ClientSession kwargs)。"""
    log = log if log is not None else []
    captured: dict[str, Any] = {}

    class ClientSession:
        def __init__(self, read_stream, write_stream, **kwargs):
            captured["streams"] = (read_stream, write_stream)
            captured["kwargs"] = kwargs
            captured["instance"] = session

        async def __aenter__(self):
            log.append("enter:session")
            if session is None:
                raise AssertionError("stub ClientSession 没有配置 session 对象")
            return session

        async def __aexit__(self, *_exc):
            log.append("exit:session")
            return False

    mcp = ModuleType("mcp")
    mcp.ClientSession = ClientSession
    client_mod = ModuleType("mcp.client")
    stdio_mod = ModuleType("mcp.client.stdio")
    sse_mod = ModuleType("mcp.client.sse")

    def _stdio_client(params):
        log.append("stdio_client")
        return stdio_cm if stdio_cm is not None else _CM("stdio", log)

    def _sse_client(url, timeout=None):
        log.append(f"sse_client:{url}:{timeout}")
        return sse_cm if sse_cm is not None else _CM("sse", log)

    stdio_mod.StdioServerParameters = _StdioServerParameters
    stdio_mod.stdio_client = _stdio_client
    sse_mod.sse_client = _sse_client
    mcp.client = client_mod
    client_mod.stdio = stdio_mod
    client_mod.sse = sse_mod

    for name, mod in [
        ("mcp", mcp),
        ("mcp.client", client_mod),
        ("mcp.client.stdio", stdio_mod),
        ("mcp.client.sse", sse_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return log, captured


def _remove_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """确保 ``import mcp`` 失败，模拟"没装可选依赖"的部署。"""
    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mcp", None)


class _FakeSession:
    def __init__(self, *, tools=None, init_error=None, result=None):
        self.tools = tools or []
        self.init_error = init_error
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self):
        if self.init_error is not None:
            raise self.init_error
        return SimpleNamespace(capabilities={})

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def _cfg(**overrides: Any) -> MCPServerConfig:
    base: dict[str, Any] = {
        "name": "catalog",
        "transport": "stdio",
        "command": "python3",
        "args": ("-m", "srv"),
        "allowed_tools": ("get_sku",),
        "risk_level": RiskLevel.LOW,
        "timeout_seconds": 3.0,
    }
    base.update(overrides)
    return MCPServerConfig(**base)


# ---------------------------------------------------------------------------
# 懒加载：没装 mcp 时必须降级而不是炸掉整个进程
# ---------------------------------------------------------------------------


class TestLazyDependency:
    async def test_missing_sdk_raises_unavailable_not_import_error(self, monkeypatch):
        _remove_sdk(monkeypatch)
        client = McpSdkClient(_cfg())
        with pytest.raises(MCPUnavailableError):
            await client.connect()

    async def test_missing_sdk_error_names_the_optional_dependency(self, monkeypatch):
        _remove_sdk(monkeypatch)
        client = McpSdkClient(_cfg())
        with pytest.raises(MCPUnavailableError) as ei:
            await client.connect()
        # 错误文案要能直接指导排障：说清是可选依赖没装，而不是"内部错误"。
        assert "mcp" in str(ei.value).lower()
        assert "ImportError" not in str(ei.value)


# ---------------------------------------------------------------------------
# transport 选择与前置校验
# ---------------------------------------------------------------------------


class TestTransportSelection:
    async def test_stdio_requires_a_command(self, monkeypatch):
        _install_stub_sdk(monkeypatch)
        with pytest.raises(MCPConfigurationError):
            await McpSdkClient(_cfg(command="")).connect()

    async def test_sse_requires_a_url(self, monkeypatch):
        _install_stub_sdk(monkeypatch)
        with pytest.raises(MCPConfigurationError):
            await McpSdkClient(_cfg(transport="sse", url="")).connect()

    async def test_stdio_passes_command_and_args_to_the_sdk(self, monkeypatch):
        seen: dict[str, Any] = {}

        class _Params:
            def __init__(self, command, args):
                seen["command"] = command
                seen["args"] = args

        log, _ = _install_stub_sdk(monkeypatch, session=_FakeSession())
        monkeypatch.setattr(
            "mcp.client.stdio.StdioServerParameters",
            _Params,
            raising=False,
        )
        monkeypatch.setattr(
            "mcp.client.stdio.stdio_client",
            lambda params: _CM("stdio", log),
            raising=False,
        )
        await McpSdkClient(_cfg()).connect()
        assert seen == {"command": "python3", "args": ["-m", "srv"]}

    async def test_sse_clients_url_with_a_bounded_timeout(self, monkeypatch):
        log, _ = _install_stub_sdk(monkeypatch, session=_FakeSession())
        await McpSdkClient(_cfg(transport="sse", url="http://127.0.0.1:9/sse")).connect()
        # 握手超时上限 5s：server 配置的 3s 更小，取小者。
        assert "sse_client:http://127.0.0.1:9/sse:3.0" in log


# ---------------------------------------------------------------------------
# read timeout：SDK 1.x 收 timedelta，2.x 收 float
# ---------------------------------------------------------------------------


class TestSdkReadTimeout:
    def test_timedelta_signature_gets_a_timedelta(self, monkeypatch):
        _install_stub_sdk(monkeypatch)

        class ClientSession:
            def __init__(self, *a, read_timeout_seconds: timedelta = None, **k):
                pass

        monkeypatch.setattr("mcp.ClientSession", ClientSession, raising=False)
        value = McpSdkClient(_cfg(timeout_seconds=3.0))._sdk_read_timeout()
        assert isinstance(value, timedelta)
        # 余量必须严格大于外层超时，否则谁先到期取决于调度时序。
        assert value.total_seconds() == pytest.approx(6.0)

    def test_float_signature_gets_a_float(self, monkeypatch):
        _install_stub_sdk(monkeypatch)

        class ClientSession:
            def __init__(self, *a, read_timeout_seconds: float = None, **k):
                pass

        monkeypatch.setattr("mcp.ClientSession", ClientSession, raising=False)
        value = McpSdkClient(_cfg(timeout_seconds=3.0))._sdk_read_timeout()
        assert isinstance(value, float)
        assert value == pytest.approx(6.0)

    def test_unannotated_signature_gets_a_float(self, monkeypatch):
        _install_stub_sdk(monkeypatch)

        class ClientSession:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr("mcp.ClientSession", ClientSession, raising=False)
        assert McpSdkClient(_cfg(timeout_seconds=3.0))._sdk_read_timeout() == pytest.approx(6.0)

    async def test_session_receives_the_slack_multiplied_read_timeout(self, monkeypatch):
        log, captured = _install_stub_sdk(monkeypatch, session=_FakeSession())
        await McpSdkClient(_cfg(timeout_seconds=3.0)).connect()
        assert captured["kwargs"]["read_timeout_seconds"] == pytest.approx(6.0)

    async def test_streams_are_handed_to_the_session_in_order(self, monkeypatch):
        log, captured = _install_stub_sdk(monkeypatch, session=_FakeSession())
        await McpSdkClient(_cfg()).connect()
        assert captured["streams"] == ("read-stdio", "write-stdio")


# ---------------------------------------------------------------------------
# connect：幂等 / 握手失败分类 / 关闭顺序
# ---------------------------------------------------------------------------


class TestConnect:
    async def test_connect_is_idempotent(self, monkeypatch):
        log, _ = _install_stub_sdk(monkeypatch, session=_FakeSession())
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.connect()
        assert log.count("enter:session") == 1

    async def test_handshake_timeout_becomes_mcp_timeout(self, monkeypatch):
        _install_stub_sdk(
            monkeypatch, session=_FakeSession(init_error=TimeoutError()), stdio_cm=None
        )
        client = McpSdkClient(_cfg(timeout_seconds=0.05))
        with pytest.raises(MCPTimeoutError):
            await client.connect()

    async def test_handshake_failure_becomes_unavailable(self, monkeypatch):
        _install_stub_sdk(monkeypatch, session=_FakeSession(init_error=RuntimeError("boom")))
        with pytest.raises(MCPUnavailableError) as ei:
            await McpSdkClient(_cfg()).connect()
        assert "RuntimeError" in str(ei.value)

    async def test_transport_enter_failure_is_classified_not_leaked(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(
            monkeypatch,
            session=_FakeSession(),
            stdio_cm=_CM("stdio", log, enter_error=OSError("no such binary")),
        )
        with pytest.raises(MCPUnavailableError):
            await McpSdkClient(_cfg(command="definitely-missing-binary")).connect()

    async def test_failed_connect_tears_the_transport_down(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(
            monkeypatch,
            session=_FakeSession(init_error=RuntimeError("boom")),
            stdio_cm=_CM("stdio", log),
        )
        client = McpSdkClient(_cfg())
        with pytest.raises(MCPUnavailableError):
            await client.connect()
        # 半开连接必须被回收，否则 stdio 子进程会泄漏。
        assert "exit:stdio" in log
        assert client._session is None
        assert client._session_cm is None
        assert client._transport_cm is None


# ---------------------------------------------------------------------------
# close：逆序、幂等、best-effort
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_exits_session_before_transport(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(monkeypatch, session=_FakeSession(), stdio_cm=_CM("stdio", log), log=log)
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.close()
        assert log.index("exit:session") < log.index("exit:stdio")

    async def test_close_is_idempotent(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(monkeypatch, session=_FakeSession(), stdio_cm=_CM("stdio", log), log=log)
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.close()
        await client.close()
        assert log.count("exit:stdio") == 1

    async def test_close_without_connect_is_a_noop(self, monkeypatch):
        _install_stub_sdk(monkeypatch)
        await McpSdkClient(_cfg()).close()

    async def test_teardown_exception_does_not_break_close(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(
            monkeypatch,
            session=_FakeSession(),
            stdio_cm=_CM("stdio", log, exit_error=OSError("gone")),
        )
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.close()
        # best-effort：即便 stdio 回收失败，状态也必须复位。
        assert client._session_cm is None
        assert client._transport_cm is None
        assert client._session is None

    async def test_reconnect_after_close_gets_a_fresh_session(self, monkeypatch):
        log: list[str] = []
        _install_stub_sdk(monkeypatch, session=_FakeSession(), stdio_cm=_CM("stdio", log), log=log)
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.close()
        await client.connect()
        assert log.count("enter:session") == 2


# ---------------------------------------------------------------------------
# list_tools / call_tool
# ---------------------------------------------------------------------------


class TestSessionDelegation:
    async def test_list_tools_returns_the_tools_list(self, monkeypatch):
        sentinel = object()
        _install_stub_sdk(monkeypatch, session=_FakeSession(tools=[sentinel]))
        assert await McpSdkClient(_cfg()).list_tools() == [sentinel]

    async def test_list_tools_of_a_result_without_tools_is_empty(self, monkeypatch):
        _install_stub_sdk(monkeypatch, session=_FakeSession())
        # 畸形响应不得让 list_tools 崩在属性访问上。
        assert await McpSdkClient(_cfg()).list_tools() == []

    async def test_call_tool_forwards_name_and_arguments(self, monkeypatch):
        session = _FakeSession(result="ok")
        _install_stub_sdk(monkeypatch, session=session)
        assert await McpSdkClient(_cfg()).call_tool("get_sku", {"sku": "1"}) == "ok"
        assert session.calls == [("get_sku", {"sku": "1"})]

    async def test_call_tool_with_no_arguments_sends_an_empty_dict(self, monkeypatch):
        session = _FakeSession(result="ok")
        _install_stub_sdk(monkeypatch, session=session)
        await McpSdkClient(_cfg()).call_tool("get_sku", None)
        assert session.calls == [("get_sku", {})]

    async def test_delegation_connects_lazily(self, monkeypatch):
        log, _ = _install_stub_sdk(monkeypatch, session=_FakeSession())
        await McpSdkClient(_cfg()).list_tools()
        # 无需调用方显式 connect：SDK client 自己负责。
        assert "enter:session" in log


def test_no_real_asyncio_timeout_wrapper_leaks_into_transport_entering(monkeypatch):
    """防回归：stdio 传输的 __aenter__ 绝不能被 wait_for 包住。

    SDK 在 __aenter__ 时把 anyio cancel scope 绑定到当时的 task；wait_for 会把
    协程放进临时 task，导致之后 close() 抛
    ``Attempted to exit cancel scope in a different task``——每个 session 的正常
    关闭都静默失败、子进程回收退化成依赖 GC。
    """
    _install_stub_sdk(monkeypatch, session=_FakeSession())
    client = McpSdkClient(_cfg())

    async def _fail(*_a, **_k):
        raise AssertionError("transport/session 的 __aenter__ 不得被 asyncio.wait_for 包裹")

    monkeypatch.setattr(asyncio, "wait_for", _fail, raising=False)
    assert asyncio.run(client.connect()) is None


# ---------------------------------------------------------------------------
# SDK connect / close 的两个"必须吞掉/必须原样抛出"分支
# ---------------------------------------------------------------------------


class TestSdkClientErrorPassthrough:
    async def test_session_error_already_classified_is_not_re_wrapped(self, monkeypatch):
        # initialize() 抛出本模块自己的异常分类时，必须原样透传：再包一层
        # MCPUnavailableError 会把"超时"错报成"不可用"，排障方向被带偏。
        from tools.mcp_adapter import MCPTimeoutError as _Timeout

        class _Raising(_FakeSession):
            async def initialize(self):
                raise _Timeout("handshake timed out")

        _install_stub_sdk(monkeypatch, session=_Raising())
        with pytest.raises(_Timeout):
            await McpSdkClient(_cfg()).connect()

    async def test_cancellation_during_teardown_does_not_break_close(self, monkeypatch):
        # teardown 期间的取消不是调用方的 bug：让它冒出去会把"关一个 MCP session"
        # 升级成"整个进程 shutdown 失败"。CancelledError 是 BaseException，
        # 下面的 except Exception 抓不到，必须单独处理。
        log: list[str] = []
        _install_stub_sdk(
            monkeypatch,
            session=_FakeSession(),
            stdio_cm=_CM("stdio", log, exit_error=asyncio.CancelledError()),
            log=log,
        )
        client = McpSdkClient(_cfg())
        await client.connect()
        await client.close()
        assert client._session is None
        assert client._transport_cm is None
