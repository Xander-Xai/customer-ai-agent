"""MCP 工具适配层的**纯函数**契约单测（不连接任何 server）。

覆盖 PR1 落地的产品代码里所有不需要 MCP server 就能钉死的契约：

  - 工具名命名空间化（Function Calling 长度上限 + 截断不撞名）；
  - ``inputSchema`` 归一化与 fail-closed；
  - ``risk_level`` 只向上收敛（缺失 / 非法 -> HIGH，绝不因配置笔误放行）；
  - ``MCP_SERVERS`` allowlist 解析（整体非法 / 单条非法 / transport 白名单）；
  - 启动期结构校验（``validate_mcp_settings``）；
  - ToolRegistry 的 ``source`` 可观测性维度（不改变执行语义）。

**不在这里测什么**：跨进程 / 传输 / 策略 / 时序的端到端取证属于 PR2，
见 ``tests/integration/test_mcp_contract_e2e.py`` 与
``tests/integration/fake_mcp_server.py``。本文件**不引入** MCP server，
因此不需要 ``mcp`` SDK 即可运行。
"""

from __future__ import annotations

import contextlib
import json
import logging

import pytest

from core.config import validate_mcp_settings
from core.hitl.risk import RiskLevel
from tools.mcp_adapter import (
    DEFAULT_RISK_LEVEL,
    FC_NAME_MAX_LEN,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    MCP_NAME_PREFIX,
    SUPPORTED_PRIMITIVE_TYPES,
    SUPPORTED_SCHEMA_KEYWORDS,
    MCPConfigurationError,
    MCPInvalidSchemaError,
    MCPServerConfig,
    MCPToolAdapter,
    build_mcp_adapters,
    load_mcp_server_configs,
    normalize_input_schema,
    qualified_tool_name,
    register_mcp_tools,
)
from tools.tool_registry import SOURCE_MCP, SOURCE_NATIVE, ToolRegistry

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Tool name namespacing
# ---------------------------------------------------------------------------


class TestQualifiedToolName:
    def test_prefix_is_namespaced_and_stable(self):
        assert qualified_tool_name("catalog", "get_sku") == "mcp__catalog__get_sku"

    def test_illegal_characters_are_sanitized(self):
        name = qualified_tool_name("cat alog", "get/sku")
        assert name == "mcp__cat_alog__get_sku"
        assert all(c.isalnum() or c in "_-" for c in name)

    def test_short_name_is_not_truncated(self):
        assert len(qualified_tool_name("c", "t")) <= FC_NAME_MAX_LEN

    def test_long_name_is_capped_at_function_calling_limit(self):
        name = qualified_tool_name("s" * 40, "t" * 40)
        assert len(name) == FC_NAME_MAX_LEN

    def test_truncation_does_not_collide_across_distinct_tools(self):
        # 截断是「先截断再补 sha256 前 8 位」，因此不同的 (server, tool)
        # 即使共享同一段前缀也不会撞名 —— 否则会静默覆盖掉一个已注册的工具。
        a = qualified_tool_name("s" * 40, "prefix_" + "a" * 40)
        b = qualified_tool_name("s" * 40, "prefix_" + "b" * 40)
        assert a != b

    def test_same_input_is_deterministic(self):
        assert qualified_tool_name("catalog", "get_sku") == qualified_tool_name(
            "catalog", "get_sku"
        )


# ---------------------------------------------------------------------------
# inputSchema normalization (fail closed)
# ---------------------------------------------------------------------------


class TestNormalizeInputSchema:
    def test_missing_schema_becomes_empty_object_schema(self):
        assert normalize_input_schema(None) == {"type": "object", "properties": {}}

    def test_type_defaults_to_object(self):
        out = normalize_input_schema({"properties": {"a": {"type": "string"}}})
        assert out["type"] == "object"
        assert out["properties"] == {"a": {"type": "string"}}

    def test_missing_properties_defaults_to_empty_dict(self):
        assert normalize_input_schema({"type": "object"})["properties"] == {}

    def test_non_object_type_fails_closed(self):
        # 非 object 的 schema 无法保证 Function Calling 侧安全 → 不静默猜测。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"type": "string"})

    def test_non_dict_schema_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(["not", "a", "dict"])

    def test_non_dict_properties_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"type": "object", "properties": ["a"]})


# ---------------------------------------------------------------------------
# nested inputSchema 递归校验（PR #42）
#
# 修复前的缺口：`normalize_input_schema` 只看**顶层**的 object/properties，
# 于是 nested 畸形 schema 会原样放行，最终经 `ToolRegistry.get_openai_tools()`
# 进入 LLM 请求的 `tools[].function.parameters`。provider 对该字段的校验是
# **全有或全无**的 —— 一处畸形拒收整条请求，所有 native 工具一起陪葬。
# 下面每个用例都锁住"畸形绝不进注册表"。
# ---------------------------------------------------------------------------


class TestNestedSchemaFailsClosed:
    def test_property_subschema_that_is_not_an_object_fails_closed(self):
        # 任务书里的原始 case：顶层完全合法，只有子 schema 是裸字符串。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"type": "object", "properties": {"x": "invalid"}})

    @pytest.mark.parametrize("bad", [42, True, None, [], "invalid", 3.5])
    def test_every_non_object_subschema_fails_closed(self, bad):
        # 注意 None / [] 与"缺失"不是一回事：显式给出畸形值必须拒绝，
        # 而不是像顶层 properties 那样被补成默认值。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"type": "object", "properties": {"x": bad}})

    def test_nested_object_subschema_is_validated_recursively(self):
        # 畸形藏在第二层：只看一层的话 `outer` 是个合法 object 就放行了。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(
                {
                    "type": "object",
                    "properties": {"outer": {"type": "object", "properties": {"inner": 42}}},
                }
            )

    def test_nested_properties_that_is_not_an_object_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"o": {"type": "object", "properties": ["a"]}}})

    def test_nested_properties_explicit_null_fails_closed(self):
        # 顶层 `properties: null` 会被补成 {}（宽松归一化），但**嵌套**的
        # `properties: null` 没有任何合法解释，必须拒绝。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"o": {"type": "object", "properties": None}}})

    def test_malformation_is_caught_at_arbitrary_depth(self):
        # 递归的意义：校验必须一路走到叶子，而不是"逐层各看一层"。
        node: dict = {"type": "object", "properties": {"c": "invalid"}}
        for _ in range(4):
            node = {"type": "object", "properties": {"n": node}}
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(node)

    def test_valid_deeply_nested_schema_is_accepted(self):
        # 反向对照：合法的深层结构不能被误杀，否则真实工具会被静默丢掉。
        node: dict = {"type": "string", "description": "leaf"}
        for _ in range(4):
            node = {"type": "object", "properties": {"n": node}, "required": ["n"]}
        assert normalize_input_schema(node)["type"] == "object"


class TestArrayItemsValidation:
    def test_items_that_is_not_an_object_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"xs": {"type": "array", "items": "invalid"}}})

    def test_items_subschema_is_validated_recursively(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(
                {"properties": {"xs": {"type": "array", "items": {"type": "not-a-type"}}}}
            )

    def test_items_may_itself_be_an_array_of_objects(self):
        out = normalize_input_schema(
            {
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"a": {"type": "string"}}},
                    }
                },
                "required": ["rows"],
            }
        )
        assert out["properties"]["rows"]["items"]["type"] == "object"

    def test_array_without_items_is_accepted(self):
        # `{"type": "array"}` 是合法 JSON Schema（items 不受限），不是畸形。
        assert normalize_input_schema({"properties": {"xs": {"type": "array"}}})["properties"]

    def test_malformed_items_inside_a_nested_array_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(
                {
                    "properties": {
                        "groups": {
                            "type": "array",
                            "items": {"type": "array", "items": "invalid"},
                        }
                    }
                }
            )


class TestRequiredValidation:
    @pytest.mark.parametrize("bad", ["a", 1, True, None, {"a": True}])
    def test_required_must_be_an_array(self, bad):
        # 任务书点名的第二个 case。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"a": {"type": "string"}}, "required": bad})

    @pytest.mark.parametrize("bad", [["missing"], ["a", "a"], [1], [""], [None], [["a"]]])
    def test_malformed_required_entries_fail_closed(self, bad):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"a": {"type": "string"}}, "required": bad})

    def test_required_referring_to_unknown_field_fails_closed(self):
        # 引用不存在的字段名 → schema 永远无法被满足（或被 provider 拒收），
        # 两种结局都不是"可用的归一化工具"。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(
                {"properties": {"a": {"type": "string"}}, "required": ["a", "ghost"]}
            )

    def test_nested_required_is_validated_against_its_own_properties(self):
        # `required` 必须对着**同级** properties 解析：内层的 ghost 不能
        # 因为外层有同名字段就被放行。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(
                {
                    "properties": {
                        "ghost": {"type": "string"},
                        "o": {"type": "object", "required": ["ghost"]},
                    }
                }
            )

    def test_valid_required_is_preserved(self):
        out = normalize_input_schema(
            {
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            }
        )
        assert out["required"] == ["a"]


class TestPrimitiveTypeValidation:
    @pytest.mark.parametrize("t", ["string", "number", "integer", "boolean", "null"])
    def test_supported_primitives_are_preserved(self, t):
        assert (
            normalize_input_schema({"properties": {"x": {"type": t}}})["properties"]["x"]["type"]
            == t
        )

    @pytest.mark.parametrize(
        "bad", ["str", "String", "object ", "", "any", 1, True, None, [], {}, "dict"]
    )
    def test_unsupported_types_fail_closed(self, bad):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"x": {"type": bad}}})

    def test_type_union_is_supported(self):
        # `["string", "null"]` 在 MCP 生态里很常见（"可为 null"），
        # 支持它只是多一个元素循环，不构成实现 JSON Schema engine。
        out = normalize_input_schema({"properties": {"x": {"type": ["string", "null"]}}})
        assert out["properties"]["x"]["type"] == ["string", "null"]

    def test_empty_type_union_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"x": {"type": []}}})

    def test_type_union_with_unsupported_member_fails_closed(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"x": {"type": ["string", "any"]}}})


class TestSupportedSchemaSubsetIsClosed:
    def test_whitelist_is_exactly_the_documented_subset(self):
        # 显式钉住子集边界：往白名单里加关键字必须是**有意识的**决定，
        # 而不是"顺手让某个 server 的 schema 通过"。
        assert sorted(SUPPORTED_SCHEMA_KEYWORDS) == [
            "description",
            "items",
            "properties",
            "required",
            "type",
        ]

    def test_supported_types_are_exactly_the_documented_set(self):
        assert sorted(SUPPORTED_PRIMITIVE_TYPES) == [
            "array",
            "boolean",
            "integer",
            "null",
            "number",
            "object",
            "string",
        ]

    @pytest.mark.parametrize(
        "kw",
        [
            "enum",
            "const",
            "oneOf",
            "anyOf",
            "allOf",
            "not",
            "$ref",
            "$defs",
            "definitions",
            "patternProperties",
            "additionalProperties",
            "format",
            "default",
            "minimum",
            "maximum",
            "minLength",
            "pattern",
            "nullable",
            "title",
            "examples",
        ],
    )
    def test_keywords_outside_the_subset_fail_closed(self, kw):
        # 这些关键字**没有任何本地逻辑会去解释**。放行等于把一段从未被校验过
        # 的语义原样塞进 provider 请求 —— 与本模块 fail closed 的立场相反。
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"x": {"type": "string", kw: "v"}}})

    @pytest.mark.parametrize("kw", ["enum", "$ref", "additionalProperties"])
    def test_unsupported_keyword_fails_closed_even_at_top_level(self, kw):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"type": "object", kw: "v"})

    def test_description_must_be_a_string(self):
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": {"x": {"description": 42}}})


class TestSchemaValidationBounds:
    """递归校验面对的是不受信任的远端输入，本身必须有界。"""

    def test_nesting_deeper_than_the_limit_fails_closed(self):
        node: dict = {"type": "string"}
        for _ in range(MAX_SCHEMA_DEPTH + 2):
            node = {"type": "object", "properties": {"n": node}}
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema(node)

    def test_nesting_within_the_limit_is_accepted(self):
        node: dict = {"type": "string"}
        for _ in range(MAX_SCHEMA_DEPTH - 3):
            node = {"type": "object", "properties": {"n": node}}
        assert normalize_input_schema(node)["type"] == "object"

    def test_too_many_properties_fail_closed(self):
        # 深度上限管不住"宽"：十万个扁平 properties 深度只有 1。
        # 而 max_payload_bytes 只约束调用 payload，不约束 list_tools 的描述符。
        props = {f"p{i}": {"type": "string"} for i in range(MAX_SCHEMA_NODES + 10)}
        with pytest.raises(MCPInvalidSchemaError):
            normalize_input_schema({"properties": props})

    def test_wide_but_within_budget_is_accepted(self):
        props = {f"p{i}": {"type": "string"} for i in range(50)}
        assert len(normalize_input_schema({"properties": props})["properties"]) == 50


# ---------------------------------------------------------------------------
# Fail closed 发生在 tool registration **之前**
# ---------------------------------------------------------------------------


class _FakeMcpClient:
    """最小 MCP client：只实现 ``MCPClientProtocol`` 要求的三个方法。"""

    def __init__(self, tools: list[dict]):
        self._tools = tools

    async def list_tools(self) -> list[dict]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return {"content": [{"text": "ok"}]}

    async def close(self) -> None:
        return None


class TestMalformedSchemaNeverReachesRegistration:
    def _adapter(self, tools: list[dict]) -> MCPToolAdapter:
        config = MCPServerConfig(
            name="cat", allowed_tools=("good", "bad"), risk_level=RiskLevel.LOW
        )
        return MCPToolAdapter(config, client=_FakeMcpClient(tools))

    async def test_nested_malformed_tool_is_skipped_and_good_one_still_registers(self):
        adapter = self._adapter(
            [
                {
                    "name": "good",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
                # 顶层合法、nested 畸形：修复前会被注册进 FC payload。
                {"name": "bad", "inputSchema": {"type": "object", "properties": {"x": "invalid"}}},
            ]
        )
        registry = ToolRegistry()
        assert await register_mcp_tools(registry, adapter) == ["mcp__cat__good"]

        # 真正的断言在这里：畸形工具**没有**出现在 provider 会看到的 payload 里。
        payload = registry.get_openai_tools()
        assert [t["function"]["name"] for t in payload] == ["mcp__cat__good"]

    async def test_required_not_an_array_tool_never_reaches_the_payload(self):
        adapter = self._adapter(
            [
                {
                    "name": "bad",
                    "inputSchema": {"properties": {"q": {"type": "string"}}, "required": "q"},
                }
            ]
        )
        registry = ToolRegistry()
        assert await register_mcp_tools(registry, adapter) == []
        assert registry.get_openai_tools() == []

    async def test_valid_nested_schema_reaches_the_payload_verbatim(self):
        schema = {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "description": "上限"},
                    },
                    "required": ["tags"],
                }
            },
            "required": ["filter"],
        }
        adapter = self._adapter([{"name": "good", "inputSchema": schema}])
        registry = ToolRegistry()
        assert await register_mcp_tools(registry, adapter) == ["mcp__cat__good"]
        params = registry.get_openai_tools()[0]["function"]["parameters"]
        assert params["properties"]["filter"]["properties"]["tags"]["items"] == {"type": "string"}
        assert params["required"] == ["filter"]


# ---------------------------------------------------------------------------
# Risk policy: converge upward only
# ---------------------------------------------------------------------------


class TestRiskLevelCoercion:
    def test_default_is_high(self):
        assert DEFAULT_RISK_LEVEL is RiskLevel.HIGH

    def test_explicit_low_is_honoured(self):
        cfg = load_mcp_server_configs(_servers(risk_level="low"))
        assert cfg[0].risk_level is RiskLevel.LOW

    def test_whitespace_and_case_are_normalized(self):
        cfg = load_mcp_server_configs(_servers(risk_level="  LoW "))
        assert cfg[0].risk_level is RiskLevel.LOW

    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_missing_risk_level_converges_to_high(self, bad):
        raw = json.dumps([{"name": "s", "risk_level": bad}])
        cfg = load_mcp_server_configs(raw)
        assert cfg[0].risk_level is RiskLevel.HIGH

    @pytest.mark.parametrize("bad", ["read", "write", "safe", "lowest", 1, True])
    def test_illegal_risk_level_never_converges_downward(self, bad):
        # 关键安全性质：非法值（含历史词汇 read/write）只向上收敛到 HIGH，
        # 绝不 fail-open 到 LOW/MEDIUM。
        raw = json.dumps([{"name": "s", "risk_level": bad}])
        cfg = load_mcp_server_configs(raw)
        assert cfg[0].risk_level is RiskLevel.HIGH

    def test_medium_is_preserved_and_is_not_low(self):
        cfg = load_mcp_server_configs(_servers(risk_level="medium"))
        assert cfg[0].risk_level is RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Allowlist parsing
# ---------------------------------------------------------------------------


def _servers(**overrides):
    item = {
        "name": "catalog",
        "transport": "stdio",
        "command": "python3",
        "args": ["-m", "srv"],
        "allowed_tools": ["get_sku"],
        "risk_level": "low",
    }
    item.update(overrides)
    return json.dumps([item])


@contextlib.contextmanager
def _capture_mcp_warnings():
    """直接挂在 ``tools.mcp`` logger 上收集 WARNING 记录。

    ``core.logger.get_logger`` 统一设置 ``propagate=False``（避免日志重复输出到
    root），因此 pytest 的 ``caplog``（挂在 root handler 上）看不到这些记录 ——
    与 ``test_erp_authorization.py`` / ``test_cache_cross_user_isolation.py``
    同因同解。
    """
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    handler.setLevel(logging.WARNING)
    target = logging.getLogger("tools.mcp")
    target.addHandler(handler)
    try:
        yield records
    finally:
        target.removeHandler(handler)


class TestLoadMcpServerConfigs:
    def test_valid_allowlist_is_parsed(self):
        cfg = load_mcp_server_configs(_servers())
        assert len(cfg) == 1
        assert cfg[0].name == "catalog"
        assert cfg[0].transport == "stdio"
        assert cfg[0].allowed_tools == ("get_sku",)

    def test_unparsable_json_yields_no_servers(self):
        # 整体不可解析 -> []（禁用 MCP），绝不"尽力而为"连接未声明的 server。
        assert load_mcp_server_configs("{not json") == []
        assert load_mcp_server_configs("") == []

    def test_non_array_json_yields_no_servers(self):
        assert load_mcp_server_configs('{"name": "catalog"}') == []

    def test_entry_without_name_is_skipped(self):
        raw = json.dumps([{"transport": "stdio", "command": "python3"}])
        assert load_mcp_server_configs(raw) == []

    def test_unknown_transport_is_rejected(self):
        raw = json.dumps(
            [
                {
                    "name": "catalog",
                    "transport": "carrier-pigeon",
                    "command": "python3",
                    "risk_level": "low",
                }
            ]
        )
        assert load_mcp_server_configs(raw) == []

    @pytest.mark.parametrize("transport", ["stdio", "sse"])
    def test_supported_transports_are_allowed(self, transport):
        raw = json.dumps(
            [
                {
                    "name": "catalog",
                    "transport": transport,
                    "command": "python3",
                    "url": "https://example.invalid/mcp",
                    "risk_level": "low",
                }
            ]
        )
        cfg = load_mcp_server_configs(raw)
        assert cfg[0].transport == transport

    def test_empty_allowed_tools_list_means_nothing_is_allowed(self):
        # 空 allowlist != 全部允许：这是 allowlist 语义的关键方向。
        cfg = load_mcp_server_configs(_servers(allowed_tools=[]))
        assert cfg[0].allowed_tools == ()

    def test_server_name_with_double_underscore_is_rejected(self):
        # `mcp__{server}__{tool}` 的分隔符就是 `__`，server 名里出现 `__`
        # 会让命名空间可被伪造。
        raw = json.dumps(
            [
                {
                    "name": "cat__alog",
                    "transport": "stdio",
                    "command": "python3",
                    "risk_level": "low",
                }
            ]
        )
        assert load_mcp_server_configs(raw) == []

    def test_defaults_are_applied_when_omitted(self):
        raw = json.dumps([{"name": "s", "transport": "stdio", "command": "x"}])
        cfg = load_mcp_server_configs(raw, default_timeout=7.5, default_max_payload=1234)
        assert cfg[0].timeout_seconds == 7.5
        assert cfg[0].max_payload_bytes == 1234

    def test_invalid_negative_timeout_fails_closed(self):
        raw = json.dumps(
            [{"name": "s", "transport": "stdio", "command": "x", "timeout_seconds": -1}]
        )
        assert load_mcp_server_configs(raw) == []


class TestBuildMcpAdapters:
    def test_disabled_server_is_skipped(self):
        cfg = load_mcp_server_configs(_servers(enabled=False))
        assert build_mcp_adapters(cfg) == []

    def test_enabled_server_yields_one_adapter(self):
        cfg = load_mcp_server_configs(_servers(enabled=True))
        assert len(build_mcp_adapters(cfg)) == 1


# ---------------------------------------------------------------------------
# ``enabled`` 类型安全：只接受 JSON boolean，绝不真值转换
# ---------------------------------------------------------------------------


class TestEnabledIsStrictlyBoolean:
    @pytest.mark.parametrize("flag", [True, False])
    def test_json_boolean_is_honoured(self, flag):
        cfg = load_mcp_server_configs(_servers(enabled=flag))
        assert cfg[0].enabled is flag
        # 反向断言：禁用的条目留在 configs 里（可观测），只是不建 adapter。
        assert len(build_mcp_adapters(cfg)) == (1 if flag else 0)

    def test_omitted_enabled_keeps_existing_default(self):
        # 全局开关是另一个旋钮（MCP_ENABLED）；allowlist 内省略 enabled 仍是启用。
        cfg = load_mcp_server_configs(_servers())
        assert cfg[0].enabled is True

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param("false", id="str-false"),
            pytest.param("true", id="str-true"),
            pytest.param("False", id="str-False-cap"),
            pytest.param("no", id="str-no"),
            pytest.param("", id="str-empty"),
            pytest.param("  ", id="str-blank"),
            pytest.param(1, id="int-1"),
            pytest.param(0, id="int-0"),
            pytest.param(-1, id="int-negative"),
            pytest.param(1.0, id="float-1.0"),
            pytest.param(None, id="null"),
            pytest.param([], id="empty-list"),
            pytest.param([False], id="list-false"),
            pytest.param(["false"], id="list-str-false"),
            pytest.param({}, id="empty-dict"),
            pytest.param({"enabled": False}, id="dict-nested"),
        ],
    )
    def test_non_boolean_enabled_is_never_coerced(self, bad):
        # 关键安全性质：``bool(value)`` 是真值判断而非类型判断，
        # ``bool("false") is True``——沿用它会让"想关掉"变成"启用"。
        # 非 boolean 一律 fail closed，整条记录被跳过（不再进入 configs）。
        cfg = load_mcp_server_configs(_servers(enabled=bad))
        assert cfg == []
        assert build_mcp_adapters(cfg) == []

    @pytest.mark.parametrize(
        "truthy_but_invalid",
        [
            pytest.param("false", id="str-false"),
            pytest.param(1, id="int-1"),
            pytest.param([], id="empty-list"),
            pytest.param({}, id="empty-dict"),
        ],
    )
    def test_truthy_non_boolean_does_not_enable(self, truthy_but_invalid):
        # 上一条的反向断言：这些值在真值语义下全是 True。若实现回退到
        # ``bool(...)``，这里会得到 enabled=True + 1 个 adapter。
        cfg = load_mcp_server_configs(_servers(enabled=truthy_but_invalid))
        assert all(c.enabled is not True for c in cfg)

    def test_invalid_enabled_skips_only_the_offending_entry(self):
        # 单条笔误不该让整个 allowlist 失效（既有契约：跳过 + 记日志）。
        raw = json.dumps(
            [
                {"name": "broken", "transport": "stdio", "command": "x", "enabled": "false"},
                {"name": "healthy", "transport": "stdio", "command": "x", "risk_level": "low"},
            ]
        )
        cfg = load_mcp_server_configs(raw)
        assert [c.name for c in cfg] == ["healthy"]
        assert len(build_mcp_adapters(cfg)) == 1

    def test_warning_identifies_the_offending_server(self):
        # 错误信息必须能定位到 server：只有 server 名能对上 MCP_SERVERS
        # 里那条出问题的记录，运维才知道该改哪一行。
        raw = json.dumps(
            [{"name": "billing-prod", "transport": "stdio", "command": "x", "enabled": "false"}]
        )
        with _capture_mcp_warnings() as records:
            assert load_mcp_server_configs(raw) == []
        messages = "\n".join(r.getMessage() for r in records)
        assert "billing-prod" in messages
        assert "enabled" in messages

    @pytest.mark.parametrize(
        ("bad", "type_name"),
        [
            pytest.param("false", "str", id="str-false"),
            pytest.param(1, "int", id="int-1"),
            pytest.param(None, "NoneType", id="null"),
            pytest.param([], "list", id="empty-list"),
            pytest.param({}, "dict", id="empty-dict"),
        ],
    )
    def test_warning_reports_value_and_python_type(self, bad, type_name):
        # 排障需要看到"写进去的是什么、解析成了什么类型"，而不只是"不合法"。
        raw = json.dumps([{"name": "srv", "transport": "stdio", "command": "x", "enabled": bad}])
        with _capture_mcp_warnings() as records:
            load_mcp_server_configs(raw)
        messages = "\n".join(r.getMessage() for r in records)
        assert "srv" in messages
        assert type_name in messages

    def test_valid_enabled_emits_no_enabled_warning(self):
        # 反向断言：合法 boolean 不该产生噪音告警。
        with _capture_mcp_warnings() as records:
            load_mcp_server_configs(_servers(enabled=True))
            load_mcp_server_configs(_servers(enabled=False))
        assert [r for r in records if "enabled" in r.getMessage()] == []

    def test_other_fields_are_not_touched_by_the_enabled_check(self):
        # enabled 的严格化不得改变其他字段的解析语义。
        cfg = load_mcp_server_configs(
            _servers(
                enabled=True,
                transport="sse",
                url="https://example.invalid/mcp",
                args=["--x"],
                allowed_tools=["a", "b"],
                timeout_seconds=9.5,
                max_payload_bytes=2048,
                risk_level="medium",
            )
        )[0]
        assert cfg.transport == "sse"
        assert cfg.url == "https://example.invalid/mcp"
        assert cfg.args == ("--x",)
        assert cfg.allowed_tools == ("a", "b")
        assert cfg.timeout_seconds == 9.5
        assert cfg.max_payload_bytes == 2048
        assert cfg.risk_level is RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Startup-time structural validation
# ---------------------------------------------------------------------------


class TestValidateMcpSettings:
    def test_disabled_needs_no_allowlist(self):
        assert validate_mcp_settings(enabled=False, servers="") == []

    def test_enabled_without_allowlist_is_rejected(self):
        errors = validate_mcp_settings(enabled=True, servers="")
        assert errors and "allowlist" in errors[0]

    def test_unparsable_json_is_rejected(self):
        assert validate_mcp_settings(enabled=True, servers="{nope")

    def test_non_array_is_rejected(self):
        assert validate_mcp_settings(enabled=True, servers='{"name":"a"}')

    def test_empty_array_is_rejected(self):
        assert validate_mcp_settings(enabled=True, servers="[]")

    def test_entry_without_name_is_rejected(self):
        assert validate_mcp_settings(enabled=True, servers='[{"transport":"stdio"}]')

    def test_missing_risk_level_is_reported_at_startup(self):
        errors = validate_mcp_settings(
            enabled=True, servers='[{"name":"a","transport":"stdio","command":"x"}]'
        )
        assert any("risk_level" in e for e in errors)

    def test_illegal_risk_level_is_reported_at_startup(self):
        errors = validate_mcp_settings(
            enabled=True,
            servers='[{"name":"a","transport":"stdio","command":"x","risk_level":"read"}]',
        )
        assert any("risk_level" in e for e in errors)

    def test_valid_allowlist_passes(self):
        assert (
            validate_mcp_settings(
                enabled=True,
                servers=(
                    '[{"name":"a","transport":"stdio","command":"x",'
                    '"allowed_tools":["t"],"risk_level":"low"}]'
                ),
            )
            == []
        )

    def test_non_positive_payload_cap_is_rejected(self):
        errors = validate_mcp_settings(enabled=True, servers=_servers(), max_payload_bytes=0)
        assert any("MCP_MAX_PAYLOAD_BYTES" in e for e in errors)


# ---------------------------------------------------------------------------
# ToolRegistry source is observability-only
# ---------------------------------------------------------------------------


class TestRegistrySourceIsObservabilityOnly:
    def test_default_source_is_native(self):
        reg = ToolRegistry()
        reg.register(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: {"ok": True},
        )
        assert reg.source_for("t") == SOURCE_NATIVE
        assert reg.tools_by_source(SOURCE_MCP) == []

    def test_explicit_mcp_source_is_recorded(self):
        reg = ToolRegistry()
        reg.register(
            name="mcp__catalog__get_sku",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: {"ok": True},
            risk_level="low",
            source=SOURCE_MCP,
        )
        assert reg.source_for("mcp__catalog__get_sku") == SOURCE_MCP
        assert reg.tools_by_source(SOURCE_MCP) == ["mcp__catalog__get_sku"]

    def test_unknown_tool_has_no_source(self):
        assert ToolRegistry().source_for("absent") is None

    def test_unknown_source_yields_empty_list(self):
        assert ToolRegistry().tools_by_source("carrier-pigeon") == []

    async def test_source_does_not_change_execution_semantics(self):
        # `source` 纯粹是可观测性维度：同样的 handler + risk_level，
        # native 与 mcp 来源的执行结果必须一致（不因来源而改写行为）。
        results = []
        for source in (SOURCE_NATIVE, SOURCE_MCP):
            reg = ToolRegistry()
            reg.register(
                name="t",
                description="d",
                parameters={"type": "object", "properties": {}},
                handler=lambda args: {"echo": args["v"]},
                risk_level="low",
                source=source,
            )
            results.append(await reg.execute("t", {"v": 7}))
        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Error taxonomy is reachable and distinct
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    def test_configuration_error_is_exported(self):
        assert issubclass(MCPConfigurationError, Exception)

    def test_name_prefix_constant_is_the_documented_one(self):
        assert MCP_NAME_PREFIX == "mcp__"
