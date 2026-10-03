"""``tools/mcp_adapter.py`` 的**运行时**契约（PR #42 覆盖率修复）。

与既有 ``tests/unit/test_mcp_adapter.py`` 的分工：

- 既有文件覆盖**纯函数**契约（名称命名空间化 / schema 归一化 / risk_level 收敛 /
  allowlist 解析 / 启动期校验），不需要任何 client；
- 本文件覆盖**适配器运行时**：``connect`` / ``discover_tools`` / ``invoke`` /
  结果归一化 / ``register_mcp_tools`` 与错误分类。这些路径此前**一行都没有测试**
  —— ``tools/mcp_adapter.py`` 在 PR #42 落地 418 条语句、其中 254 条未被执行，
  把整体 coverage 从 80%+ 拖到 79.84%，使 CI 的 ``fail_under = 80`` 变红。

驱动方式用模块已有的注入缝（``MCPToolAdapter(config, client=...)`` +
``MCPClientProtocol``），**不引入新的产品代码**，也不需要真实 MCP server 或官方
SDK：被测的是本仓自己的策略逻辑（allowlist / fail-closed / 错误分类 / payload
上限 / 结果归一化），不是第三方 SDK 的行为。

SDK client 侧（``McpSdkClient``）单列在 ``test_mcp_sdk_client.py``：它需要 ``mcp``
包，而 CI 的 ``test`` job 只装 ``requirements.txt``、``mcp`` 在
``requirements-optional.txt``，故用 stub 而非真实依赖。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from core.hitl.risk import RiskLevel
from core.tool_result_cache import ToolCachePolicy
from tools.mcp_adapter import (
    _MISSING,
    MCPConfigurationError,
    MCPError,
    MCPPayloadTooLargeError,
    MCPServerConfig,
    MCPTimeoutError,
    MCPToolAdapter,
    MCPToolExecutionError,
    MCPUnauthorizedToolError,
    MCPUnavailableError,
    _read_field,
    load_mcp_server_configs,
    register_mcp_tools,
)
from tools.tool_registry import SOURCE_MCP, ToolRegistry

# ---------------------------------------------------------------------------
# 夹具：MCP SDK 1.x（camelCase wire 字段）形态
# ---------------------------------------------------------------------------


class _Tool:
    """工具描述符。``schema=...`` / ``description=...`` 可表示"字段完全不存在"。"""

    _ABSENT = object()

    def __init__(self, name: Any, schema: Any = _ABSENT, description: Any = "d"):
        self.name = name
        if schema is not self._ABSENT:
            self.inputSchema = schema
        if description is not self._ABSENT:
            self.description = description


class _Content:
    def __init__(self, text: str):
        self.text = text


class _Result:
    """``CallToolResult``（SDK 1.x camelCase）。

    ``is_error`` 默认 ``False``：真实 SDK 的 ``CallToolResult.isError`` 恒存在
    （默认 False）。需要表达"字段整个缺失"时显式传 ``_ABSENT``。
    """

    _ABSENT = object()

    def __init__(
        self,
        *,
        content: Any = None,
        is_error: Any = False,
        structured: Any = _ABSENT,
    ):
        if content is not None:
            self.content = content
        if is_error is not self._ABSENT:
            self.isError = is_error
        if structured is not self._ABSENT:
            self.structuredContent = structured


class _ResultWithoutIsError:
    """畸形结果：有 content，但 ``isError`` / ``is_error`` 两个名字都认不出。"""

    def __init__(self, content: Any = None):
        if content is not None:
            self.content = content


class _FakeClient:
    """满足 ``MCPClientProtocol`` 的确定性 client。

    刻意**不实现** ``connect``：该方法在 ``MCPClientProtocol`` 里是可选的，
    适配器用 ``getattr`` 探测，因此用到这里的用例同时锁住"没有 connect 的 client
    也能正常工作"。
    """

    def __init__(self, tools=None, result=None, *, list_error=None, call_error=None):
        self.tools = tools if tools is not None else []
        self.result = result
        self.list_error = list_error
        self.call_error = call_error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = 0

    async def list_tools(self) -> list[Any]:
        if self.list_error is not None:
            raise self.list_error
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return self.result

    async def close(self) -> None:
        self.closed += 1


def _cfg(**overrides: Any) -> MCPServerConfig:
    base: dict[str, Any] = {
        "name": "catalog",
        "transport": "stdio",
        "command": "python3",
        "allowed_tools": ("get_sku",),
        "risk_level": RiskLevel.LOW,
        "timeout_seconds": 5.0,
    }
    base.update(overrides)
    return MCPServerConfig(**base)


def _ok(text: str = "sku-1") -> _Result:
    return _Result(content=[_Content(text)], is_error=False)


def _payload_size(arguments: dict[str, Any]) -> int:
    """与 ``_validate_payload`` 完全相同的编码方式，用于边界断言。"""
    return len(json.dumps(arguments, ensure_ascii=False, default=str).encode("utf-8"))


# ---------------------------------------------------------------------------
# _read_field：camelCase / snake_case / dict 三种形态 + 缺失哨兵
# ---------------------------------------------------------------------------


class _BothDictAndAttr(dict):
    """同时具备 dict 键与同名属性的载荷。"""

    name = "from-attribute"


class TestReadFieldDualNaming:
    """SDK 1.x camelCase 与 2.x snake_case 必须都能读到，且"缺失"要与 None 区分。"""

    def test_camel_case_attribute_is_read(self):
        assert _read_field(_Tool("t", schema={"type": "object"}), "input_schema") == {
            "type": "object"
        }

    def test_snake_case_attribute_is_read(self):
        class V2:
            input_schema = {"type": "object"}

        assert _read_field(V2(), "input_schema") == {"type": "object"}

    def test_dict_payload_camel_case_is_read(self):
        assert _read_field({"inputSchema": {"type": "object"}}, "input_schema") == {
            "type": "object"
        }

    def test_dict_payload_snake_case_is_read(self):
        assert _read_field({"input_schema": {"type": "object"}}, "input_schema") == {
            "type": "object"
        }

    def test_attribute_lookup_precedes_dict_key(self):
        # 同一对象两种编码时行为必须确定：属性形态优先。
        assert _read_field(_BothDictAndAttr(name="from-dict"), "name") == "from-attribute"

    def test_absent_field_yields_the_sentinel_not_none(self):
        # 缺失的 inputSchema 说明 server 返回了畸形描述符，fail closed 跳过；
        # 值为 None 则是合法的"无参 schema"。两者必须可区分。
        assert _read_field(object(), "input_schema") is _MISSING
        assert _read_field({"inputSchema": None}, "input_schema") is None


# ---------------------------------------------------------------------------
# load_mcp_server_configs：数值字段 fail closed
# ---------------------------------------------------------------------------


class TestNumericFieldsFailClosed:
    def test_non_numeric_timeout_skips_only_that_server(self):
        raw = json.dumps(
            [
                {"name": "bad", "transport": "stdio", "command": "x", "timeout_seconds": "soon"},
                {"name": "good", "transport": "stdio", "command": "y"},
            ]
        )
        assert [c.name for c in load_mcp_server_configs(raw)] == ["good"]

    def test_non_numeric_payload_cap_skips_only_that_server(self):
        raw = json.dumps(
            [
                {
                    "name": "bad",
                    "transport": "stdio",
                    "command": "x",
                    "max_payload_bytes": {"nope": 1},
                },
                {"name": "good", "transport": "stdio", "command": "y"},
            ]
        )
        assert [c.name for c in load_mcp_server_configs(raw)] == ["good"]

    def test_numeric_strings_are_accepted(self):
        raw = json.dumps(
            [
                {
                    "name": "a",
                    "transport": "stdio",
                    "command": "x",
                    "timeout_seconds": "2.5",
                    "max_payload_bytes": "1024",
                }
            ]
        )
        cfg = load_mcp_server_configs(raw)[0]
        assert cfg.timeout_seconds == 2.5
        assert cfg.max_payload_bytes == 1024


# ---------------------------------------------------------------------------
# connect：幂等 / 超时 / 错误映射 / 注入 client 的所有权
# ---------------------------------------------------------------------------


class _ConnectClient(_FakeClient):
    def __init__(self, *, error=None, sleep: float = 0.0, result=None):
        super().__init__(result=result)
        self.error = error
        self.sleep = sleep
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.error is not None:
            raise self.error


class TestConnect:
    async def test_connect_is_idempotent(self):
        client = _ConnectClient()
        adapter = MCPToolAdapter(_cfg(), client)
        await adapter.connect()
        await adapter.connect()
        assert client.connect_calls == 1

    async def test_slow_connect_raises_timeout(self):
        adapter = MCPToolAdapter(_cfg(timeout_seconds=0.05), _ConnectClient(sleep=5.0))
        with pytest.raises(MCPTimeoutError):
            await adapter.connect()
        assert adapter._connected is False

    async def test_timed_out_connect_does_not_close_an_injected_client(self):
        # 注入的 client 由注入方代管：adapter 不得替它 close。
        client = _ConnectClient(sleep=5.0)
        adapter = MCPToolAdapter(_cfg(timeout_seconds=0.05), client)
        with pytest.raises(MCPTimeoutError):
            await adapter.connect()
        assert client.closed == 0

    async def test_refused_connection_maps_to_unavailable(self):
        adapter = MCPToolAdapter(_cfg(), _ConnectClient(error=ConnectionRefusedError("nope")))
        with pytest.raises(MCPUnavailableError):
            await adapter.connect()

    async def test_timeout_error_maps_to_timeout(self):
        adapter = MCPToolAdapter(_cfg(), _ConnectClient(error=TimeoutError()))
        with pytest.raises(MCPTimeoutError):
            await adapter.connect()

    async def test_already_classified_error_propagates_unchanged(self):
        original = MCPUnavailableError("already classified")
        adapter = MCPToolAdapter(_cfg(), _ConnectClient(error=original))
        with pytest.raises(MCPUnavailableError) as ei:
            await adapter.connect()
        assert ei.value is original

    async def test_unknown_connect_error_keeps_its_type_in_the_message(self):
        adapter = MCPToolAdapter(_cfg(), _ConnectClient(error=ValueError("weird")))
        with pytest.raises(MCPError) as ei:
            await adapter.connect()
        assert "ValueError" in str(ei.value)

    async def test_client_without_connect_attribute_still_works(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        specs = await MCPToolAdapter(_cfg(), client).discover_tools()
        assert [s.mcp_name for s in specs] == ["get_sku"]

    async def test_close_does_not_close_an_injected_client(self):
        client = _FakeClient()
        await MCPToolAdapter(_cfg(), client).close()
        assert client.closed == 0


# ---------------------------------------------------------------------------
# discover_tools：allowlist 与 schema 双重 fail closed
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    async def test_allowed_tool_is_normalized_into_a_spec(self):
        client = _FakeClient(
            tools=[
                _Tool(
                    "get_sku",
                    schema={"type": "object", "properties": {"sku": {"type": "string"}}},
                    description="lookup",
                )
            ]
        )
        specs = await MCPToolAdapter(_cfg(), client).discover_tools()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.name == "mcp__catalog__get_sku"
        assert spec.mcp_name == "get_sku"
        assert spec.description == "lookup"
        assert spec.input_schema["properties"] == {"sku": {"type": "string"}}
        assert spec.risk_level is RiskLevel.LOW
        assert spec.timeout == 5.0
        assert spec.server == "catalog"
        assert spec.source == SOURCE_MCP

    async def test_tool_outside_allowlist_is_skipped(self):
        client = _FakeClient(
            tools=[
                _Tool("get_sku", schema={"type": "object"}),
                _Tool("drop_table", schema={"type": "object"}),
            ]
        )
        specs = await MCPToolAdapter(_cfg(allowed_tools=("get_sku",)), client).discover_tools()
        assert [s.mcp_name for s in specs] == ["get_sku"]

    async def test_missing_input_schema_is_skipped_not_defaulted(self):
        # 静默注册成"无参工具"会让 LLM 凭空造参数 —— 必须 fail closed。
        client = _FakeClient(tools=[_Tool("get_sku")])
        assert await MCPToolAdapter(_cfg(), client).discover_tools() == []

    async def test_invalid_schema_type_is_skipped(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "string"})])
        assert await MCPToolAdapter(_cfg(), client).discover_tools() == []

    async def test_non_dict_properties_is_skipped(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object", "properties": []})])
        assert await MCPToolAdapter(_cfg(), client).discover_tools() == []

    @pytest.mark.parametrize("bad_name", [None, "", 123, True])
    async def test_unusable_tool_name_is_skipped(self, bad_name):
        client = _FakeClient(tools=[_Tool(bad_name)])
        assert await MCPToolAdapter(_cfg(), client).discover_tools() == []

    async def test_list_tools_failure_is_mapped_and_propagated(self):
        client = _FakeClient(list_error=ConnectionResetError("gone"))
        with pytest.raises(MCPUnavailableError):
            await MCPToolAdapter(_cfg(), client).discover_tools()

    async def test_empty_tool_list_is_not_an_error(self):
        assert await MCPToolAdapter(_cfg(), _FakeClient()).discover_tools() == []

    async def test_spec_carries_the_local_risk_level_not_a_server_claim(self):
        # server 自述的 annotations 一律不采信：风险等级只来自本地 allowlist。
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        specs = await MCPToolAdapter(_cfg(risk_level=RiskLevel.HIGH), client).discover_tools()
        assert specs[0].risk_level is RiskLevel.HIGH

    @pytest.mark.parametrize("description", [None, "", _Tool._ABSENT])
    async def test_absent_or_empty_description_becomes_empty_string(self, description):
        # 字段完全缺失时不得把内部 _MISSING 哨兵的 repr 泄进 Function Calling 上下文。
        client = _FakeClient(
            tools=[_Tool("get_sku", schema={"type": "object"}, description=description)]
        )
        specs = await MCPToolAdapter(_cfg(), client).discover_tools()
        assert specs[0].description == ""

    async def test_non_string_description_is_coerced(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"}, description=7)])
        specs = await MCPToolAdapter(_cfg(), client).discover_tools()
        assert specs[0].description == "7"


# ---------------------------------------------------------------------------
# invoke：allowlist → payload 上限 → 超时 → 错误分类
# ---------------------------------------------------------------------------


class TestInvoke:
    async def test_successful_call_returns_normalized_text(self):
        client = _FakeClient(result=_ok("sku-42"))
        out = await MCPToolAdapter(_cfg(), client).invoke("get_sku", {"sku": "42"})
        assert out == "sku-42"
        assert client.calls == [("get_sku", {"sku": "42"})]

    async def test_structured_content_wins_over_text(self):
        client = _FakeClient(
            result=_Result(content=[_Content("ignored")], structured={"sku": "42"})
        )
        assert await MCPToolAdapter(_cfg(), client).invoke("get_sku") == {"sku": "42"}

    async def test_error_result_carries_the_server_text(self):
        client = _FakeClient(result=_Result(content=[_Content("no such sku")], is_error=True))
        with pytest.raises(MCPToolExecutionError) as ei:
            await MCPToolAdapter(_cfg(), client).invoke("get_sku")
        assert str(ei.value) == "no such sku"
        assert ei.value.reason == "tool_error"

    async def test_error_result_without_text_still_raises(self):
        client = _FakeClient(result=_Result(is_error=True))
        with pytest.raises(MCPToolExecutionError) as ei:
            await MCPToolAdapter(_cfg(), client).invoke("get_sku")
        assert "MCP tool 返回错误" in str(ei.value)

    async def test_error_result_whose_content_field_is_absent_is_still_classified(self):
        # 回归：content 字段整个缺失时，_extract_text 曾去迭代 _MISSING 哨兵并抛
        # TypeError，被 invoke 的兜底 except 降级成通用 MCPError，错误分类与
        # 指标 reason 双双失真。
        client = _FakeClient(result=_Result(is_error=True))
        with pytest.raises(MCPToolExecutionError) as ei:
            await MCPToolAdapter(_cfg(), client).invoke("get_sku")
        assert ei.value.reason == "tool_error"

    async def test_success_result_whose_content_field_is_absent_yields_none(self):
        client = _FakeClient(result=_Result(is_error=False))
        assert await MCPToolAdapter(_cfg(), client).invoke("get_sku") is None

    async def test_tool_outside_allowlist_never_reaches_the_client(self):
        client = _FakeClient(result=_ok())
        with pytest.raises(MCPUnauthorizedToolError):
            await MCPToolAdapter(_cfg(), client).invoke("drop_table")
        assert client.calls == []

    async def test_empty_allowlist_permits_nothing(self):
        client = _FakeClient(result=_ok())
        with pytest.raises(MCPUnauthorizedToolError):
            await MCPToolAdapter(_cfg(allowed_tools=()), client).invoke("get_sku")

    async def test_payload_over_the_cap_is_rejected_before_dispatch(self):
        client = _FakeClient(result=_ok())
        adapter = MCPToolAdapter(_cfg(max_payload_bytes=32), client)
        with pytest.raises(MCPPayloadTooLargeError):
            await adapter.invoke("get_sku", {"sku": "x" * 200})
        assert client.calls == []

    async def test_payload_cap_counts_utf8_bytes_not_characters(self):
        # 1 个汉字 = 3 UTF-8 字节。按字符数算上限会漏放 3 倍。
        args = {"sku": "护" * 10}
        assert len(args["sku"]) == 10  # 字符数在上限之内
        assert _payload_size(args) > 16  # 字节数超限
        client = _FakeClient(result=_ok())
        with pytest.raises(MCPPayloadTooLargeError):
            await MCPToolAdapter(_cfg(max_payload_bytes=16), client).invoke("get_sku", args)

    async def test_payload_exactly_at_the_cap_is_accepted(self):
        args = {"sku": "42"}
        client = _FakeClient(result=_ok())
        adapter = MCPToolAdapter(_cfg(max_payload_bytes=_payload_size(args)), client)
        assert await adapter.invoke("get_sku", args) == "sku-1"

    async def test_circular_arguments_are_rejected_rather_than_dispatched(self):
        args: dict[str, Any] = {}
        args["self"] = args
        client = _FakeClient(result=_ok())
        with pytest.raises(MCPError):
            await MCPToolAdapter(_cfg(), client).invoke("get_sku", args)
        assert client.calls == []

    async def test_slow_call_raises_timeout(self):
        class _Slow(_FakeClient):
            async def call_tool(self, name, arguments):
                await asyncio.sleep(5.0)
                return _ok()

        adapter = MCPToolAdapter(_cfg(timeout_seconds=0.05), _Slow())
        with pytest.raises(MCPTimeoutError):
            await adapter.invoke("get_sku")

    async def test_transport_error_is_classified_as_unavailable(self):
        adapter = MCPToolAdapter(_cfg(), _FakeClient(call_error=BrokenPipeError("gone")))
        with pytest.raises(MCPUnavailableError):
            await adapter.invoke("get_sku")

    async def test_already_classified_client_error_is_not_double_wrapped(self):
        original = MCPUnauthorizedToolError("nope")
        adapter = MCPToolAdapter(_cfg(), _FakeClient(call_error=original))
        with pytest.raises(MCPUnauthorizedToolError) as ei:
            await adapter.invoke("get_sku")
        assert ei.value is original

    async def test_unknown_client_error_keeps_its_type_in_the_message(self):
        adapter = MCPToolAdapter(_cfg(), _FakeClient(call_error=ValueError("weird")))
        with pytest.raises(MCPError) as ei:
            await adapter.invoke("get_sku")
        assert "ValueError" in str(ei.value)

    async def test_arguments_default_to_empty_dict(self):
        client = _FakeClient(result=_ok())
        await MCPToolAdapter(_cfg(), client).invoke("get_sku")
        assert client.calls == [("get_sku", {})]

    async def test_caller_arguments_are_not_mutated_or_aliased(self):
        client = _FakeClient(result=_ok())
        args = {"sku": "42"}
        await MCPToolAdapter(_cfg(), client).invoke("get_sku", args)
        assert args == {"sku": "42"}

    async def test_second_call_reuses_the_established_connection(self):
        client = _ConnectClient(result=_ok())
        adapter = MCPToolAdapter(_cfg(), client)
        await adapter.invoke("get_sku")
        await adapter.invoke("get_sku")
        assert client.connect_calls == 1
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# 结果归一化（1.x camelCase / 2.x snake_case）
# ---------------------------------------------------------------------------


class _Part:
    """SDK 2.x 形态的 content part。"""

    def __init__(self, text: str):
        self.text = text


class _ResultV2:
    """SDK 2.x 形态的 ``CallToolResult``（snake_case）。"""

    def __init__(self, *, content=None, is_error=False, structured=None):
        if content is not None:
            self.content = content
        if is_error is not None:
            self.is_error = is_error
        if structured is not None:
            self.structured_content = structured


class TestNormalizeResult:
    def test_none_stays_none(self):
        assert MCPToolAdapter._normalize_result(None) is None

    def test_object_without_any_mcp_field_is_returned_verbatim(self):
        sentinel = object()
        assert MCPToolAdapter._normalize_result(sentinel) is sentinel

    def test_dict_without_any_mcp_field_is_returned_verbatim(self):
        assert MCPToolAdapter._normalize_result({"a": 1}) == {"a": 1}

    def test_content_without_a_recognisable_is_error_is_still_unreadable(self):
        # 有 content 但两个 is_error 名字都认不出 → 仍无法判定成功与否，fail closed。
        # 绝不能"看到文本就当成功"——那会让真实失败以成功返回。
        err = MCPToolAdapter._normalize_result(_ResultWithoutIsError([_Content("hi")]))
        assert isinstance(err, MCPToolExecutionError)
        assert err.reason == "unreadable_result"

    def test_snake_case_is_error_is_honoured(self):
        err = MCPToolAdapter._normalize_result(
            _ResultV2(is_error=True, content=[_Part("2.x fail")])
        )
        assert isinstance(err, MCPToolExecutionError)
        assert str(err) == "2.x fail"

    def test_snake_case_structured_content_is_honoured(self):
        result = _ResultV2(content=[_Part("ignored")], structured={"sku": "1"})
        assert MCPToolAdapter._normalize_result(result) == {"sku": "1"}

    def test_snake_case_content_parts_are_joined_in_order(self):
        result = _ResultV2(is_error=False, content=[_Part("a"), _Part("b")])
        assert MCPToolAdapter._normalize_result(result) == "a\nb"

    def test_multiple_content_parts_are_joined_in_order(self):
        assert (
            MCPToolAdapter._normalize_result(_Result(content=[_Content("a"), _Content("b")]))
            == "a\nb"
        )

    def test_content_item_without_text_falls_back_to_its_string_form(self):
        assert MCPToolAdapter._normalize_result(_Result(content=[_Content("a"), 7])) == "a\n7"

    def test_empty_content_normalizes_to_none(self):
        # 由 ToolRegistry 转成"查询完成，无结果"，不把 SDK 对象泄进上下文。
        assert MCPToolAdapter._normalize_result(_Result(content=[])) is None

    def test_none_structured_content_falls_through_to_text(self):
        assert (
            MCPToolAdapter._normalize_result(_Result(content=[_Content("fallback")])) == "fallback"
        )

    def test_is_error_false_is_success_not_failure(self):
        assert (
            MCPToolAdapter._normalize_result(_Result(is_error=False, content=[_Content("ok")]))
            == "ok"
        )

    def test_result_with_neither_is_error_nor_content_is_passed_through(self):
        # 两个字段都认不出 = 这根本不是 CallToolResult（str / 自定义对象），
        # 原样返回，由调用方自己解释；不得凭空造一个"工具失败"。
        sentinel = _ResultWithoutIsError()
        assert MCPToolAdapter._normalize_result(sentinel) is sentinel

    @pytest.mark.parametrize("content", [None, _MISSING, []])
    def test_extract_text_of_absent_content_is_empty_string(self, content):
        # 回归：_MISSING 哨兵是 truthy 的，未守卫时 ``content or []`` 会去迭代
        # 一个 object() 并抛 TypeError。
        assert MCPToolAdapter._extract_text(content) == ""


class TestMapError:
    def test_mcp_error_is_returned_unchanged(self):
        original = MCPTimeoutError("x")
        assert MCPToolAdapter._map_error(original) is original

    def test_timeout_error_maps_to_timeout(self):
        assert isinstance(MCPToolAdapter._map_error(TimeoutError()), MCPTimeoutError)

    def test_os_error_maps_to_unavailable(self):
        assert isinstance(MCPToolAdapter._map_error(OSError()), MCPUnavailableError)

    def test_connection_error_maps_to_unavailable(self):
        assert isinstance(MCPToolAdapter._map_error(ConnectionError()), MCPUnavailableError)

    def test_anything_else_maps_to_the_base_class_with_type_preserved(self):
        mapped = MCPToolAdapter._map_error(KeyError("k"))
        assert type(mapped) is MCPError
        assert "KeyError" in str(mapped)


# ---------------------------------------------------------------------------
# register_mcp_tools：只注册显式只读，且绝不覆盖 native 工具
# ---------------------------------------------------------------------------


def _registry_with_native(name: str = "mcp__catalog__get_sku") -> ToolRegistry:
    async def _native_handler(arguments: dict[str, Any]) -> str:
        return "native-result"

    reg = ToolRegistry()
    reg.register(
        name=name,
        description="native",
        parameters={"type": "object", "properties": {}},
        handler=_native_handler,
    )
    return reg


class TestRegisterMcpTools:
    async def test_low_risk_tool_is_registered(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        assert await register_mcp_tools(ToolRegistry(), MCPToolAdapter(_cfg(), client)) == [
            "mcp__catalog__get_sku"
        ]

    async def test_registered_tool_executes_through_the_adapter(self):
        client = _FakeClient(
            tools=[_Tool("get_sku", schema={"type": "object"})], result=_ok("sku-7")
        )
        registry = ToolRegistry()
        await register_mcp_tools(registry, MCPToolAdapter(_cfg(), client))
        assert await registry.execute("mcp__catalog__get_sku", {"sku": "7"}) == "sku-7"

    async def test_registered_tool_declares_explicit_read_only_semantics(self):
        # 显式 low 压过 core.hitl.risk 的金额阈值启发式：只读查询不该触发审批。
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = ToolRegistry()
        await register_mcp_tools(registry, MCPToolAdapter(_cfg(), client))
        definition = registry._tools["mcp__catalog__get_sku"]
        assert definition.risk_level == "low"
        assert definition.side_effect is False
        assert definition.source == SOURCE_MCP

    @pytest.mark.parametrize("risk", [RiskLevel.MEDIUM, RiskLevel.HIGH])
    async def test_non_low_risk_server_tools_are_not_registered(self, risk):
        # 本模块没有写操作的幂等 ledger，不假装有：非显式只读一律不注册。
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = ToolRegistry()
        assert (
            await register_mcp_tools(registry, MCPToolAdapter(_cfg(risk_level=risk), client)) == []
        )
        assert registry.list_tools() == []

    async def test_name_collision_never_overwrites_a_native_tool(self):
        # registry.register 本身是覆盖语义，MCP 侧必须先让位。
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = _registry_with_native()
        assert await register_mcp_tools(registry, MCPToolAdapter(_cfg(), client)) == []
        assert await registry.execute("mcp__catalog__get_sku", {}) == "native-result"

    async def test_cache_is_off_by_default(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = ToolRegistry()
        await register_mcp_tools(registry, MCPToolAdapter(_cfg(), client))
        policy = registry._tools["mcp__catalog__get_sku"].cache_policy
        assert policy.enabled is False
        assert policy.ttl_seconds == 0

    async def test_cache_can_be_enabled_explicitly(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = ToolRegistry()
        await register_mcp_tools(
            registry, MCPToolAdapter(_cfg(), client), enable_read_cache=True, cache_ttl_seconds=60
        )
        assert registry._tools["mcp__catalog__get_sku"].cache_policy == ToolCachePolicy(
            enabled=True, ttl_seconds=60
        )

    async def test_each_handler_binds_its_own_spec(self):
        # 经典闭包陷阱：循环里注册的 handler 必须各自绑定自己的 spec，
        # 否则所有 MCP 工具都会调用最后发现的那个工具。
        client = _FakeClient(
            tools=[
                _Tool("get_sku", schema={"type": "object"}),
                _Tool("get_name", schema={"type": "object"}),
            ],
            result=_ok("x"),
        )
        adapter = MCPToolAdapter(_cfg(allowed_tools=("get_sku", "get_name")), client)
        registry = ToolRegistry()
        assert len(await register_mcp_tools(registry, adapter)) == 2
        await registry.execute("mcp__catalog__get_name", {})
        assert client.calls[-1][0] == "get_name"

    async def test_registering_the_same_server_twice_collides_instead_of_duplicating(self):
        client = _FakeClient(tools=[_Tool("get_sku", schema={"type": "object"})])
        registry = ToolRegistry()
        adapter = MCPToolAdapter(_cfg(), client)
        await register_mcp_tools(registry, adapter)
        assert await register_mcp_tools(registry, adapter) == []
        assert registry.list_tools() == ["mcp__catalog__get_sku"]

    async def test_discover_failure_propagates_and_registers_nothing(self):
        client = _FakeClient(list_error=ConnectionResetError("gone"))
        registry = _registry_with_native(name="get_sku")
        with pytest.raises(MCPUnavailableError):
            await register_mcp_tools(registry, MCPToolAdapter(_cfg(), client))
        assert registry.list_tools() == ["get_sku"]

    async def test_server_without_discovered_tools_registers_nothing(self):
        registry = _registry_with_native(name="get_sku")
        assert await register_mcp_tools(registry, MCPToolAdapter(_cfg(), _FakeClient())) == []
        assert registry.list_tools() == ["get_sku"]


class TestErrorTaxonomyIsDistinct:
    @pytest.mark.parametrize(
        "cls",
        [
            MCPConfigurationError,
            MCPToolExecutionError,
            MCPPayloadTooLargeError,
            MCPTimeoutError,
            MCPUnauthorizedToolError,
            MCPUnavailableError,
        ],
    )
    def test_every_class_is_a_distinct_mcp_error(self, cls):
        # 调用方要能分别决策：配置错 / 工具失败 / 超限 / 超时 / 未授权 / 不可用
        # 不能互相塌缩成同一个类型。
        assert issubclass(cls, MCPError)
        siblings = [
            MCPConfigurationError,
            MCPToolExecutionError,
            MCPPayloadTooLargeError,
            MCPTimeoutError,
            MCPUnauthorizedToolError,
            MCPUnavailableError,
        ]
        for other in siblings:
            if other is not cls:
                assert not issubclass(cls, other), f"{cls.__name__} 塌缩进了 {other.__name__}"

    def test_execution_error_reason_defaults_to_tool_error(self):
        assert MCPToolExecutionError("x").reason == "tool_error"

    def test_execution_error_reason_is_overridable(self):
        assert MCPToolExecutionError("x", reason="unreadable_result").reason == "unreadable_result"


# ---------------------------------------------------------------------------
# 指标助手：任何情况下都不能打断工具调用
# ---------------------------------------------------------------------------


class TestMetricHelpersNeverBreakToolCalls:
    """``_inc`` / ``_observe`` 是 best effort：Prometheus 缺失或指标未声明时静默降级。"""

    def test_counter_without_labels(self):
        from tools.mcp_adapter import _inc

        _inc("mcp_tool_call_total")  # 无 label 形态不得抛异常

    def test_counter_with_labels(self):
        from tools.mcp_adapter import _inc

        _inc("mcp_tool_call_total", server="catalog", tool="get_sku", status="ok")

    def test_histogram_without_labels(self):
        from tools.mcp_adapter import _observe

        _observe("mcp_tool_duration_seconds", 0.01)

    def test_histogram_with_labels(self):
        from tools.mcp_adapter import _observe

        _observe("mcp_tool_duration_seconds", 0.01, server="catalog", tool="get_sku")

    @pytest.mark.parametrize(
        ("helper", "args"),
        [("_inc", ("mcp_tool_call_total",)), ("_observe", ("mcp_tool_duration_seconds", 0.1))],
    )
    def test_undeclared_metric_degrades_silently(self, monkeypatch, helper, args):
        from core import monitoring
        from tools.mcp_adapter import _inc, _observe

        monkeypatch.delattr(monitoring, args[0], raising=False)
        (_inc if helper == "_inc" else _observe)(*args)  # 不得抛异常


# ---------------------------------------------------------------------------
# 自建 client（生产路径）的所有权语义
# ---------------------------------------------------------------------------


class _SelfBuiltClient:
    """替代 ``McpSdkClient`` 的假实现，用于验证 adapter 的**所有权**语义。"""

    instances: list[_SelfBuiltClient] = []
    fail_next_connect = False
    fail_next_with_classified = False

    def __init__(self, config, *, connect_error=None):
        self.config = config
        self.connect_error = connect_error
        self.connect_calls = 0
        self.closed = 0
        _SelfBuiltClient.instances.append(self)

    async def connect(self):
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        if _SelfBuiltClient.fail_next_with_classified:
            _SelfBuiltClient.fail_next_with_classified = False
            raise MCPUnavailableError("server not listening")
        if _SelfBuiltClient.fail_next_connect:
            _SelfBuiltClient.fail_next_connect = False
            raise RuntimeError("transient handshake failure")

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        return _ok()

    async def close(self):
        self.closed += 1


@pytest.fixture
def self_built(monkeypatch):
    """把 ``tools.mcp_adapter.McpSdkClient`` 换成可控假实现（无需真实 SDK）。"""
    from tools import mcp_adapter

    _SelfBuiltClient.instances = []
    _SelfBuiltClient.fail_next_connect = False
    _SelfBuiltClient.fail_next_with_classified = False
    monkeypatch.setattr(mcp_adapter, "McpSdkClient", _SelfBuiltClient)
    return _SelfBuiltClient


class TestAdapterOwnsTheClientItCreates:
    """``MCPToolAdapter(config)``（不注入 client）是生产路径：它自己建、自己关。"""

    async def test_adapter_creates_its_own_sdk_client(self, self_built):
        adapter = MCPToolAdapter(_cfg())
        await adapter.connect()
        assert len(self_built.instances) == 1
        assert adapter._client is self_built.instances[0]
        assert adapter._owns_client is True

    async def test_owned_client_is_closed_on_close(self, self_built):
        adapter = MCPToolAdapter(_cfg())
        await adapter.connect()
        await adapter.close()
        assert self_built.instances[0].closed == 1
        assert adapter._connected is False

    async def test_already_classified_connect_error_is_raised_unchanged(self, self_built):
        # 自建 client 的 connect 抛出已分类异常时：原样透传（不重包），
        # 且不置 _connected —— 分类错误不代表连接状态变化。
        self_built.fail_next_with_classified = True
        adapter = MCPToolAdapter(_cfg())
        with pytest.raises(MCPUnavailableError) as ei:
            await adapter.connect()
        assert "server not listening" in str(ei.value)
        assert adapter._connected is False

    async def test_unclassified_connect_error_discards_and_closes_the_client(self, self_built):
        adapter = MCPToolAdapter(_cfg())
        adapter._client = _SelfBuiltClient(_cfg(), connect_error=RuntimeError("boom"))
        adapter._owns_client = True
        with pytest.raises(MCPError):
            await adapter.connect()
        # 半开 session 必须被回收（stdio 子进程不能泄漏），并允许重连。
        assert adapter._client is None
        assert adapter._connected is False

    async def test_reconnect_after_a_discarded_failure_builds_a_fresh_client(self, self_built):
        # _discard_client 的契约：置空 self._client，让下一次 connect() 拿到**全新**
        # client，而不是复用已被取消过的半开连接。（close() 不同：它保留 client
        # 包装器，由 SDK client 自己重开 session。）
        self_built.fail_next_connect = True
        adapter = MCPToolAdapter(_cfg())
        with pytest.raises(MCPError):
            await adapter.connect()
        discarded = self_built.instances[-1]
        assert discarded.closed == 1
        assert adapter._client is None

        await adapter.connect()
        assert adapter._client is not discarded
        assert adapter._client is self_built.instances[-1]
        assert adapter._connected is True

    async def test_close_keeps_the_client_wrapper_so_the_sdk_can_reopen_a_session(self, self_built):
        adapter = MCPToolAdapter(_cfg())
        await adapter.connect()
        client = adapter._client
        await adapter.close()
        await adapter.connect()
        # close() 只释放 session，不销毁 client；重建 session 由 SDK client 负责。
        assert adapter._client is client
        assert client.connect_calls == 2

    async def test_owned_client_is_closed_even_if_close_raises(self, self_built):
        adapter = MCPToolAdapter(_cfg())
        await adapter.connect()
        client = adapter._client

        async def _boom():
            raise RuntimeError("close failed")

        client.close = _boom
        await adapter.close()  # best-effort：不得冒泡
        assert adapter._connected is False

    async def test_close_without_a_client_is_a_noop(self):
        await MCPToolAdapter(_cfg()).close()
