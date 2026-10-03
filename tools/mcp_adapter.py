"""MCP（Model Context Protocol）外部工具适配器。

定位（**叠加，不是替代**）：

- **native Function Calling 保留**：ERP / RAG / 系统内建工具零额外网络跳转，
  低延迟、可注入应用级幂等、可控缓存策略、离线可测。
- **MCP 是标准化外部工具接口**：外部 / 第三方 / 跨进程 / 异构语言工具由 MCP
  server 暴露，本模块负责 discover → schema 归一化 → invoke，然后注册进
  **同一个** ``ToolRegistry``。
- Agent / ReAct / Function Calling 只看到统一的 ``ToolDefinition``
  （name / description / parameters / handler），不关心工具来源。

安全策略一律 **fail closed**（任何不确定都拒绝，而不是猜测）：

1. **server allowlist**：只连接 ``MCP_SERVERS`` 中显式声明的 server；
1b. **``enabled`` 只认 JSON boolean**：`"false"`` / `0` / `null` / `[]` / `{}` 一律
   不做真值转换，整条跳过。``bool("false") is True``，若沿用真值语义，运维想
   *关掉* 一个 server 的那次拼写错误会把它 *打开*；
2. **tool allowlist**：``allowed_tools`` 为空 = 该 server 不允许任何工具；
3. **transport 白名单**：仅 ``stdio`` / ``sse``；
4. **命名空间化**：MCP 工具名为 ``mcp__{server}__{tool}``，与 native 工具不冲突；
5. **schema 归一化 + 子集校验**：非法 ``inputSchema`` 抛错而不是静默猜测；
   归一化之后还要**逐层递归**校验嵌套结构，且只接受本仓库 Function Calling
   描述实际使用的 JSON Schema **子集**（``SUPPORTED_SCHEMA_KEYWORDS``）；
   子集之外的关键字与畸形嵌套一律 **fail closed**（见
   ``normalize_input_schema`` 的说明）；
6. **timeout / payload limit**：per-server 超时 + 调用 payload 字节上限；
7. **风险等级默认 HIGH**：``risk_level`` 缺失 / 非法一律按 ``RiskLevel.HIGH``
   （最保守）处理，绝不因配置笔误而放行，且**只向上收敛**。只有显式声明
   ``"low"`` 的 server 才会被当作只读——**只读必须显式 allow**；
8. **只注册显式只读**：MEDIUM / HIGH 的 MCP 工具一律不注册（本模块没有写操作
   幂等 ledger，不假装有）；
9. **不采信 server 自述**：故意不读 MCP ``annotations``（``readOnlyHint`` /
   ``destructiveHint``）——那是远端不受信任输入，不能覆盖本地风险策略；
10. **不另立权限体系**：MCP 只是 tool transport。注册后的工具与 native 工具共用
    ``core.hitl.risk`` 分级、HITL 审批闸门、side-effect ledger；
11. **错误分类**：超时 / 不可用 / 未授权 / 非法 schema / 超限 / 工具执行失败
    分成不同的异常类型，调用方可以分别决策。

证据边界：端到端契约证据（注册表 → 适配器 → 本地确定性 fake MCP server →
结果 → policy / telemetry）见 ``tests/integration/test_mcp_contract_e2e.py``
（``make test-mcp``）；**真实第三方 MCP server 与生产连通性是 ``NOT_VERIFIED``**
（见 docs/design/mcp-tool-adapter.md）。官方 SDK client 侧在本仓当前锁定版本
（``mcp==1.27.2``）上验证；``mcp`` 2.x 的 snake_case 字段命名由 ``_FIELD_ALIASES``
双读兼容，但 2.x **未在本仓验证**。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from core.hitl.risk import RiskLevel
from core.logger import get_logger
from core.tool_result_cache import ToolCachePolicy

from .tool_registry import SOURCE_MCP, SOURCE_NATIVE  # noqa: F401 (re-exported)

logger = get_logger("tools.mcp")

#: MCP 工具的默认风险等级：**HIGH**（最保守）。
#:
#: 本模块**不另立一套风险词汇**。历史实现曾用私有的 ``read`` / ``write`` 字符串，
#: 并把它称作与 ``low|medium|high`` 平行的"另一个维度"——但那实际上就是第二套权限
#: 体系，而且是个 fail-open：
#:
#:   1. ``MCPServerConfig.risk_level`` 默认 ``read``，于是**未配置**的 server 的工具
#:      会被注册成 ``risk_level="low"``，在 ``core.hitl.gate`` 眼里就是"低风险、
#:      不需要审批"；
#:   2. 非法值被强制回落成 ``read``——"配置写错 → 放行"；
#:   3. ``read`` / ``write`` 这个轴最终只当成一个"准不准注册"开关：从没真正到达
#:      ``core.hitl.risk``，而 ``register_mcp_tools`` 无条件把注册进来的工具写成
#:      ``low``。
#:
#: 现在统一使用 ``core.hitl.risk.RiskLevel``：缺失或非法一律按 ``HIGH`` 处理，
#: 且**只向上收敛**；只有显式声明 ``low`` 的 server 才注册——只读必须显式 allow。
DEFAULT_RISK_LEVEL = RiskLevel.HIGH

#: SDK 层 ``read_timeout_seconds`` 相对外层超时的余量倍数。
#:
#: 必须**严格大于**外层超时：外层（``MCPToolAdapter`` 里的 ``_await_with_timeout`` /
#: ``wait_for``）才是设计上的第一道防线，SDK 的读超时是第二道。若两者相等，谁先到期
#: 取决于调度时序，异常类型会在 ``MCPTimeoutError`` 与 SDK 自己的读超时异常之间
#: 摇摆 —— 把契约断言建立在一个竞态上，等于没有契约。
_SDK_READ_TIMEOUT_SLACK = 2.0

#: Function Calling 工具名的硬上限（OpenAI function name <= 64 字符）。
FC_NAME_MAX_LEN = 64
#: 命名空间前缀：Agent 侧可据此区分 native / MCP 工具。
MCP_NAME_PREFIX = "mcp__"

_SUPPORTED_TRANSPORTS = ("stdio", "sse")
# 工具名安全字符（FC 只接受 [A-Za-z0-9_-]）
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
# server 名比工具名更严格：不允许 ``__``（否则 ``mcp__{server}__{tool}`` 的
# 命名空间出现歧义），也不允许以 ``-``/``_`` 开头。
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: 字段"不存在"哨兵（与"存在但为 None"区分）。
_MISSING = object()

# ===== inputSchema 支持的 JSON Schema 子集 =====

#: 归一化之后**继续接受并逐层校验**的关键字白名单。
#:
#: 这不是完整 JSON Schema，而是本仓库 Function Calling 工具描述**实际用到**的全部
#: 词汇（见 ``tools/erp_tools.py`` / ``tools/hitl_staging_tools.py``：``type`` /
#: ``properties`` / ``required`` / ``description``，加上 MCP 生态普遍需要的
#: ``items``）。
#:
#: 白名单之外的一切（``enum`` / ``oneOf`` / ``$ref`` / ``patternProperties`` /
#: ``format`` / ``additionalProperties`` ...）一律 **fail closed** 拒绝。理由不是
#: "我们不实现完整 JSON Schema"，而是：这些关键字**没有任何本地逻辑会去解释**，
#: 放行等于把一段从未被校验过的语义原样塞进 provider 请求 —— 与本模块"任何不确定
#: 都拒绝"的基本立场相反。
#:
#: 为什么不用 ``jsonschema`` 库的 meta-schema 校验：它按完整 JSON Schema 校验，
#: 会**接受**上面这些关键字，方向正好相反；而且仓库并未声明该依赖（它只在
#: ``requirements-lock.txt`` 里作为传递依赖出现）。
SUPPORTED_SCHEMA_KEYWORDS = frozenset({"type", "description", "properties", "required", "items"})

#: ``type`` 允许的取值。
#:
#: 含 ``null``：真实 MCP server 常用它表达"该参数可为显式 null"，Function Calling
#: provider 同样接受。``object`` / ``array`` 作为子 schema 类型时，其
#: ``properties`` / ``items`` 的结构合法性由递归校验负责。
SUPPORTED_PRIMITIVE_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)

#: 嵌套深度上限。
#:
#: 递归校验面对的是**不受信任的远端输入**：没有上限时，一个刻意嵌套极深的 schema
#: 会把校验本身变成崩溃点（``RecursionError``）与 CPU DoS 面。真实工具描述的深度
#: 在个位数，12 足够宽松，同时把递归成本钉死。
MAX_SCHEMA_DEPTH = 12

#: 单个 ``inputSchema`` 内可访问的子 schema 节点数上限。
#:
#: 深度上限管不住"宽"：一个扁平但有十万个 ``properties`` 的 schema 深度只有 1。
#: 而 ``max_payload_bytes`` 只约束**调用** payload，不约束 ``list_tools`` 返回的
#: 描述符 —— 发现阶段没有任何天然体积上限，所以这里必须自带预算。
MAX_SCHEMA_NODES = 512

#: ``enabled`` 的 JSON 契约：**只接受 JSON boolean**（``true`` / ``false``）。
#:
#: 不复用 Python 真值语义，因为 ``bool(value)`` 是真值判断而不是类型判断：
#: ``bool("false")`` / ``bool([])`` / ``bool({"a": 1})`` / ``bool(1)`` 全是
#: ``True``。运维按 YAML/环境变量的肌肉记忆写 ``"enabled": "false"`` 想关掉
#: server，解析结果却是**启用**——``enabled`` 恰恰是整个配置里唯一一个语义就是
#: "关掉它"的字段，却在最常见的拼写错误下被打开。JSON 的 ``true`` / ``false``
#: 本来就会解析成 Python ``bool``，因此任何非 ``bool`` 都意味着**类型写错了**，
#: 而不是"另一种真值表示"。
#:
#: 不做字符串归一化（``"true"`` → ``True``）：那等于替运维猜意图，配置解析器
#: 一旦开始"猜"，就无法区分"故意"和"笔误"，而这里的失败方向是不可逆的连接行为。
#: 保守方向只有一个——不启用。
_ENABLED_MISSING = object()

#: MCP SDK 字段别名：1.x camelCase（wire 层）→ 2.x snake_case（Python 模型）。
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name",),
    "description": ("description",),
    "input_schema": ("inputSchema", "input_schema"),
    "content": ("content",),
    "text": ("text",),
    "is_error": ("isError", "is_error"),
    "structured_content": ("structuredContent", "structured_content"),
}


# ===== 错误分类（error taxonomy）=====


class MCPError(RuntimeError):
    """MCP 调用错误基类（所有 MCP 异常都继承它）。"""


class MCPConfigurationError(MCPError):
    """MCP 配置非法（例如 stdio 缺 command、transport 不支持）。"""


class MCPTimeoutError(MCPError):
    """MCP 连接或调用超时。"""


class MCPUnauthorizedToolError(MCPError):
    """工具不在 allowlist 中（fail closed）。"""


class MCPUnavailableError(MCPError):
    """MCP server 不可用 / 连接失败。"""


class MCPInvalidSchemaError(MCPError):
    """MCP tool schema 非法（无法归一化为 JSON Schema object）。"""


class MCPPayloadTooLargeError(MCPError):
    """调用 payload 超过配置上限（发送前拒绝）。"""


class MCPToolExecutionError(MCPError):
    """MCP server 正常返回了 ``isError`` / ``is_error`` 结果（工具自身执行失败）。

    ``reason`` 区分两种"不是成功"：``tool_error`` = server 明确说工具失败；
    ``unreadable_result`` = 结果里**既认不出** ``isError`` 也认不出
    ``is_error``（畸形 ``CallToolResult``），无法判定调用是否成功。
    两者都是 fail closed，但排障方向完全不同，所以指标要能分开。
    """

    def __init__(self, message: str, *, reason: str = "tool_error"):
        super().__init__(message)
        self.reason = reason


# ===== 配置（server allowlist）=====


@dataclass(frozen=True)
class MCPServerConfig:
    """单个 MCP server 的 allowlist 配置（来自 ``MCP_SERVERS`` JSON）。

    ``allowed_tools`` 为空表示**不允许任何工具**（fail closed），
    不是"允许全部"。
    """

    name: str
    transport: str = "stdio"  # stdio | sse
    url: str = ""  # sse
    command: str = ""  # stdio
    args: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    timeout_seconds: float = 15.0
    max_payload_bytes: int = 32768
    enabled: bool = True
    """是否启用该 server（只有 JSON boolean ``true`` / ``false`` 合法）。

    JSON 解析由 ``load_mcp_server_configs`` 严格把关：string / number / null /
    list / dict 一律不算 boolean，整条记录被跳过（fail closed），绝不隐式转换
    ——``bool("false") is True`` 会让"想关掉"变成"启用"。详见 ``_ENABLED_MISSING``。
    """
    risk_level: RiskLevel = DEFAULT_RISK_LEVEL
    """该 server 全部工具的本地风险等级，**默认 HIGH**（未配置即最保守）。

    必须是 ``RiskLevel`` 成员：``__post_init__`` 与 ``load_mcp_server_configs``
    负责把任意输入收敛到合法值，且非法输入只会**向上**收敛到 HIGH，绝不向下。
    """

    def __post_init__(self) -> None:
        # dataclass 不做类型校验，``risk_level`` 完全可能被写成裸字符串（测试夹具、
        # 未来的调用方）。这里统一收敛一次，避免下游 ``is not RiskLevel.LOW`` 这种
        # **身份比较**在传入 "low" 时静默失效——``RiskLevel`` 是 ``str`` Enum，
        # ``"low" == RiskLevel.LOW`` 成立但二者不是同一个对象，工具会被误判成
        # "非只读"而全部跳过。收敛方向依然只向上。
        object.__setattr__(
            self, "risk_level", _coerce_risk_level(self.risk_level, server=self.name)
        )


@dataclass
class MCPToolSpec:
    """归一化后的 MCP 工具描述（尚未进入 ``ToolRegistry``）。"""

    name: str  # 命名空间化后的统一名称
    mcp_name: str  # server 侧的原始工具名
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel
    timeout: float
    source: str = SOURCE_MCP
    server: str = ""


class MCPClientProtocol(Protocol):
    """最小 MCP client 接口（官方 SDK 实现、fake 实现都满足它）。

    ``connect`` / ``disconnect`` 是可选的：适配器用 ``getattr`` 探测，
    因此只有 ``list_tools`` / ``call_tool`` / ``close`` 的 client 也能工作。
    """

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    async def close(self) -> None: ...


# ===== 名称与 schema 归一化 =====


def qualified_tool_name(server: str, tool: str, *, max_len: int = FC_NAME_MAX_LEN) -> str:
    """把 MCP 工具名命名空间化为 ``mcp__{server}__{tool}``。

    非 ``[A-Za-z0-9_-]`` 字符被替换为 ``_``；超长时截断并追加 raw 名的
    sha256 前 8 位，保证不同 ``(server, tool)`` 不会因截断而撞名，
    且始终满足 Function Calling 的长度上限。
    """
    raw = f"{MCP_NAME_PREFIX}{server}__{tool}"
    safe = _NAME_RE.sub("_", raw)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[: max_len - 9]}_{digest}"


def _validate_type(declared: Any, *, path: str) -> None:
    """校验 ``type``：单个受支持的类型名，或非空的受支持类型名数组。

    数组形式（``["string", "null"]``）在 MCP 生态里很常见（表达"可为 null"），
    结构上只是多一个元素循环，因此支持它不构成"实现 JSON Schema engine"；
    但元素仍必须逐一落在 :data:`SUPPORTED_PRIMITIVE_TYPES` 内。
    """
    if isinstance(declared, list):
        if not declared:
            raise MCPInvalidSchemaError(f"{path}.type 是空数组")
        names: list[Any] = list(declared)
    else:
        names = [declared]
    for name in names:
        # bool 是 int 的子类，但这里比的是字符串集合，True 不会误命中；
        # 保留显式 isinstance 是为了让"类型名必须是字符串"这条意图可读。
        if not isinstance(name, str) or name not in SUPPORTED_PRIMITIVE_TYPES:
            raise MCPInvalidSchemaError(
                f"{path}.type 非法: {name!r}（支持: {sorted(SUPPORTED_PRIMITIVE_TYPES)}）"
            )


def _validate_required(node: dict[str, Any], *, path: str) -> None:
    """校验 ``required``：必须是**元素唯一、且都能在同级 ``properties`` 里找到**的字符串数组。

    "元素能在 ``properties`` 里找到"这条是这里唯一带语义判断的检查，值得单独说：
    ``required`` 引用一个不存在的字段名时，这个 schema 要么永远无法被满足
    （对 LLM 而言等于"这个工具怎么调都不对"），要么被 provider 直接拒收。
    两种结局都不是"归一化后的可用工具"，所以按畸形处理而不是原样放行。
    """
    required = node["required"]
    if not isinstance(required, list):
        raise MCPInvalidSchemaError(f"{path}.required 不是 array: {type(required).__name__}")
    props = node.get("properties")
    known: set[str] = set(props) if isinstance(props, dict) else set()
    seen: set[str] = set()
    for entry in required:
        if not isinstance(entry, str) or not entry:
            raise MCPInvalidSchemaError(f"{path}.required 元素非法: {entry!r}")
        if entry in seen:
            raise MCPInvalidSchemaError(f"{path}.required 元素重复: {entry!r}")
        if entry not in known:
            raise MCPInvalidSchemaError(
                f"{path}.required 引用了同级 properties 中不存在的字段: {entry!r}"
            )
        seen.add(entry)


def _validate_input_schema_subset(schema: dict[str, Any]) -> None:
    """递归校验 ``inputSchema`` 是否落在 :data:`SUPPORTED_SCHEMA_KEYWORDS` 子集内。

    为什么必须在**注册进 ToolRegistry 之前**做：``ToolDefinition.parameters`` 会被
    ``ToolRegistry.get_openai_tools()`` **原样**塞进 LLM 请求的
    ``tools[].function.parameters``，中间没有任何一层会再校验它。于是一个 nested
    畸形 schema 的后果不是"那个工具不好用"，而是 **provider 拒收整条请求** ——
    所有 native 工具一起陪葬。所以畸形必须在本地、在注册之前拦住。

    遍历用闭包持有 ``remaining`` 预算：递归里逐层传一个计数器会让每个辅助函数的
    签名都被预算污染，而"深度 / 节点数"这两个上限本来就只属于这一次遍历。
    """
    remaining = MAX_SCHEMA_NODES

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal remaining

        if depth > MAX_SCHEMA_DEPTH:
            raise MCPInvalidSchemaError(f"{path} 嵌套深度超过上限 {MAX_SCHEMA_DEPTH}")
        remaining -= 1
        if remaining < 0:
            raise MCPInvalidSchemaError(f"inputSchema 子 schema 节点数超过上限 {MAX_SCHEMA_NODES}")

        if not isinstance(node, dict):
            raise MCPInvalidSchemaError(f"{path} 不是 JSON object: {type(node).__name__}")

        unknown = sorted(set(node) - SUPPORTED_SCHEMA_KEYWORDS)
        if unknown:
            raise MCPInvalidSchemaError(
                f"{path} 含子集之外的关键字 {unknown}（支持: {sorted(SUPPORTED_SCHEMA_KEYWORDS)}）"
            )

        if "type" in node:
            _validate_type(node["type"], path=path)
        if "description" in node and not isinstance(node["description"], str):
            raise MCPInvalidSchemaError(f"{path}.description 不是 string")
        if "required" in node:
            _validate_required(node, path=path)

        props = node.get("properties", _MISSING)
        if props is not _MISSING:
            if not isinstance(props, dict):
                raise MCPInvalidSchemaError(f"{path}.properties 不是 object")
            for name, sub in props.items():
                if not isinstance(name, str) or not name:
                    raise MCPInvalidSchemaError(f"{path}.properties 键非法: {name!r}")
                walk(sub, f"{path}.properties.{name}", depth + 1)

        items = node.get("items", _MISSING)
        if items is not _MISSING:
            walk(items, f"{path}.items", depth + 1)

    walk(schema, "inputSchema", 0)


def normalize_input_schema(raw: Any) -> dict[str, Any]:
    """把 MCP ``inputSchema`` 归一化为 **受支持子集内**的 JSON Schema object。

    顶层归一化（宽松）：

    - ``None`` → 最小可用的空 object schema；
    - 缺 ``type`` 补 ``object``，缺 ``properties`` 补 ``{}``；
    - 非 object 的 ``type`` / 非 dict 的 ``properties`` / 非 dict 的 schema
      全部 **fail closed** 抛 ``MCPInvalidSchemaError``（不静默猜测）。

    随后**逐层递归**校验嵌套结构（:func:`_validate_input_schema_subset`）。这一步
    不宽松，因为 provider 对 ``tools[].function.parameters`` 的校验是**全有或全无**的：
    任何一层的畸形都会让整条请求被拒，而不是只让那个工具失效。因此"顶层看着正常"
    不足以构成放行理由 —— 典型的漏网形态是
    ``{"properties": {"x": "invalid"}}``（子 schema 不是 object）与
    ``{"required": "x"}``（``required`` 不是 array）。

    本函数是远端 ``inputSchema`` 进入 ``ToolDefinition.parameters`` 的**唯一**通道
    （``discover_tools`` → ``register_mcp_tools``），所以把 fail closed 放在这里，
    结构上就不存在绕过路径。
    """
    if raw is None:
        return {"type": "object", "properties": {}}
    if not isinstance(raw, dict):
        raise MCPInvalidSchemaError("inputSchema 不是 JSON object")
    schema = dict(raw)
    declared = schema.get("type")
    if declared not in (None, "object"):
        raise MCPInvalidSchemaError(f"inputSchema.type 非法: {declared!r}")
    schema["type"] = "object"
    props = schema.get("properties")
    if props is None:
        schema["properties"] = {}
    elif not isinstance(props, dict):
        raise MCPInvalidSchemaError("inputSchema.properties 不是 object")
    _validate_input_schema_subset(schema)
    return schema


def _read_field(raw: Any, logical: str) -> Any:
    """读取 MCP 工具/结果字段，同时兼容 SDK 的两套命名。

    MCP SDK 1.x 的 Python 模型沿用 wire 层 camelCase
    （``Tool.inputSchema`` / ``CallToolResult.isError`` / ``structuredContent``），
    2.x 改为 snake_case（``input_schema`` / ``is_error`` / ``structured_content``）。
    只认其中一套会让"读不到字段"退化成静默降级（例如 schema 变成空 object，
    错误结果被当成成功），所以这里枚举两套别名，并且同时支持 dict 形态。

    字段完全不存在时返回 ``_MISSING``，与"存在但为 None"区分开 ——
    缺失的 ``inputSchema`` 说明 server 返回了畸形描述符，fail closed 跳过该工具。
    """
    for attr in _FIELD_ALIASES[logical]:
        value = getattr(raw, attr, _MISSING)
        if value is not _MISSING:
            return value
    if isinstance(raw, dict):
        for attr in _FIELD_ALIASES[logical]:
            if attr in raw:
                return raw[attr]
    return _MISSING


def _coerce_risk_level(raw: Any, *, server: str = "") -> RiskLevel:
    """把任意 ``risk_level`` 配置收敛成合法 ``RiskLevel``，**只向上不向下**。

    - 未配置（``None`` / 空串） -> ``HIGH``：MCP 工具默认按最保守策略处理。
      想要"只读"必须**显式**写 ``"low"``，否则该 server 的工具不会被注册。
    - 合法值 -> 原样返回（大小写 / 空白归一化）。
    - 非法值（含历史词汇 ``read`` / ``write``）-> ``HIGH`` + 告警。

    绝不把非法值收敛到 LOW/MEDIUM：那是"配置写错 → 放行"方向的 fail-open。
    """
    where = f"MCP server {server} " if server else ""
    if isinstance(raw, RiskLevel):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        logger.warning(
            "%srisk_level 未配置，按最保守的 %s 处理（该 server 的工具不会被注册；"
            '确认只读请显式配置 risk_level="low"）',
            where,
            DEFAULT_RISK_LEVEL.value,
        )
        return DEFAULT_RISK_LEVEL
    try:
        return RiskLevel(str(raw).strip().lower())
    except ValueError:
        logger.warning(
            "%srisk_level 非法（%r），按最保守的 %s 处理（绝不因配置错误而放行）",
            where,
            raw,
            DEFAULT_RISK_LEVEL.value,
        )
        return DEFAULT_RISK_LEVEL


def _coerce_enabled_flag(item: dict[str, Any], *, name: str) -> Any:
    """读取 ``enabled``，只接受 JSON boolean；类型非法时返回 ``_ENABLED_MISSING``。

    返回值三态（而不是 bool / raise）：**存在且合法** → ``True`` / ``False``；
    **键不存在** → 沿用既有默认（``True``，即"进了 allowlist 就是启用的"，
    全局开关是另一个旋钮 ``MCP_ENABLED``，此处不改）；**存在但不是
    boolean** → ``_ENABLED_MISSING``，由调用方跳过整条。

    为什么非法时"跳过"而不是"收敛到 False"：两者都让 server 不被连接
    （净效果一致），但保留条目会造出一个**无法与运维主动禁用区分**的状态——
    排障时看到 ``enabled=False`` 却找不到任何告警线索，而配置里明明写着一个
    类型错误的字段。跳过让告警成为该次故障的唯一、且不会被误读的解释。

    为什么不用异常 / 整体 reject：``load_mcp_server_configs`` 对单条非法记录的
    既定契约就是"跳过 + 记日志"（``transport`` / 数值字段同此，见
    ``core.config.validate_mcp_settings`` 的说明），一条 server 的笔误不该让
    整个 allowlist 一起失效。可观测性由 container 兜底：若所有条目都被跳过，
    ``MCP_FAIL_CLOSED=true``（生产）会以 ``ConfigurationError`` 拒绝启动，
    不会"配错了却静默跑成 native-only"。
    """
    if "enabled" not in item:
        return True
    raw = item["enabled"]
    # bool 必须在 int 之前判断：``isinstance(True, int)`` 为 True。
    if isinstance(raw, bool):
        return raw
    logger.warning(
        "MCP server %s 的 enabled 非法（%r，%s）——只接受 JSON boolean "
        'true / false；字符串 "false"、数字 0/1、null、数组、对象都不接受'
        '（真值语义会把 "false" 判成启用）。该 server 已按 fail closed 跳过，'
        "不会被连接。",
        name,
        raw,
        type(raw).__name__,
    )
    return _ENABLED_MISSING


def load_mcp_server_configs(
    raw: str,
    *,
    default_timeout: float = 15.0,
    default_max_payload: int = 32768,
) -> list[MCPServerConfig]:
    """解析 ``MCP_SERVERS`` JSON allowlist。

    任何非法输入都 **fail closed**：整体不可解析时返回 ``[]``（禁用 MCP），
    单条非法时只跳过该条并记录原因。绝不"尽力而为"地连接未声明的 server。
    """
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("MCP_SERVERS JSON 解析失败，禁用 MCP: %s", type(e).__name__)
        return []
    if not isinstance(data, list):
        logger.warning("MCP_SERVERS 必须是 JSON 数组，禁用 MCP")
        return []

    configs: list[MCPServerConfig] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            logger.warning("MCP server 配置缺少 name，跳过")
            continue
        name = str(item["name"])
        if not _SERVER_NAME_RE.match(name) or "__" in name:
            logger.warning(
                "MCP server 名称非法（仅 [A-Za-z0-9_-]、不含 __、不以分隔符开头），跳过: %s", name
            )
            continue
        # enabled 先于 transport / 数值校验：它决定这条记录**是否成立**
        # （名字与合法性是"准入"，transport 与超时是"怎么连"）。一个类型写错的
        # enabled 会让后面的连接细节失去意义，先报它对排障更直接。
        enabled = _coerce_enabled_flag(item, name=name)
        if enabled is _ENABLED_MISSING:
            continue
        transport = str(item.get("transport", "stdio")).strip().lower()
        if transport not in _SUPPORTED_TRANSPORTS:
            logger.warning("MCP server %s transport 不支持: %s", name, transport)
            continue
        try:
            timeout = float(item.get("timeout_seconds", default_timeout))
            max_payload = int(item.get("max_payload_bytes", default_max_payload))
        except (TypeError, ValueError):
            logger.warning("MCP server %s 配置数值非法，跳过", name)
            continue
        if timeout <= 0 or max_payload <= 0:
            # 0 / 负值会让"每次调用都超时"或"任何 payload 都超限"，
            # 那是不可用的配置而不是可用的限制。跳过该 server。
            logger.warning(
                "MCP server %s 的 timeout_seconds / max_payload_bytes 必须为正数，跳过", name
            )
            continue
        risk_level = _coerce_risk_level(item.get("risk_level"), server=name)
        configs.append(
            MCPServerConfig(
                name=name,
                transport=transport,
                url=str(item.get("url", "")),
                command=str(item.get("command", "")),
                args=tuple(str(a) for a in item.get("args", []) or []),
                allowed_tools=tuple(str(t) for t in item.get("allowed_tools", []) or []),
                timeout_seconds=timeout,
                max_payload_bytes=max_payload,
                enabled=enabled,
                risk_level=risk_level,
            )
        )
    return configs


# ===== 指标（best effort：Prometheus 缺失时静默降级）=====


async def _await_with_timeout(awaitable: Any, seconds: float) -> Any:
    """给「注入进来的」client connect 加超时。

    这条路径**只用于注入的 client**（测试 fake 等），它们不涉及 anyio cancel
    scope，所以 ``asyncio.wait_for`` 在这里是安全的。

    真实 SDK client 走的是 ``McpSdkClient.connect``，那里**刻意不用**
    ``asyncio.wait_for`` 包裹 ``__aenter__``：``wait_for`` 会把协程放进临时 task，
    而 SDK 的 stdio 传输在 ``__aenter__`` 时把 anyio cancel scope 绑定到当时的
    task，导致之后 ``close()`` 抛
    ``RuntimeError: Attempted to exit cancel scope in a different task than it was
    entered in`` —— 每个 session 的正常关闭都静默失败、子进程回收退化成依赖 GC。
    ``anyio.fail_after`` 包住 context manager 的进入同样会破坏 cancel scope 栈
    （实测 ``Attempted to exit a cancel scope that isn't the current task's current
    cancel scope``），所以那里只对握手阶段加超时。
    """
    return await asyncio.wait_for(awaitable, timeout=seconds)


def _inc(name: str, **labels: Any) -> None:
    try:
        from core import monitoring

        metric = getattr(monitoring, name, None)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).inc()
        else:
            metric.inc()
    except Exception:  # pragma: no cover - 指标永远不能打断工具调用
        pass


def _observe(name: str, value: float, **labels: Any) -> None:
    try:
        from core import monitoring

        metric = getattr(monitoring, name, None)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    except Exception:  # pragma: no cover - 指标永远不能打断工具调用
        pass


class MCPToolAdapter:
    """单个 MCP server 的适配器：discover / normalize / invoke。

    生命周期由调用方管理：``connect``（或首次 discover/invoke 时惰性连接）
    → 使用 → ``close``。
    """

    def __init__(self, config: MCPServerConfig, client: MCPClientProtocol | None = None):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._connected = False

    async def connect(self) -> None:
        """建立 MCP session（幂等）。

        两条路径的超时语义**故意不同**，原因是 anyio cancel scope 的绑定规则
        （详见 ``_await_with_timeout`` 的说明）：

        - **自建 SDK client**：不加外层包装，交给 ``McpSdkClient.connect`` 在
          当前 task 内自行给握手加 ``anyio.fail_after``。加了外层包装会让随后的
          ``close()`` 在任何环境下都抛 cancel scope RuntimeError。
        - **注入的 client**：套 ``asyncio.wait_for``，给不自带超时的实现兜底。
        """
        if self._connected:
            return
        if self._client is None:
            self._client = McpSdkClient(self.config)
            try:
                await self._client.connect()
            except MCPError:
                self._connected = False
                raise
            except Exception as e:
                await self._discard_client()
                raise self._map_error(e) from e
            self._connected = True
            return

        connect = getattr(self._client, "connect", None)
        if connect is not None:
            try:
                await _await_with_timeout(connect(), self.config.timeout_seconds)
            except asyncio.TimeoutError as e:
                # wait_for 取消的是内层协程，client 自己没机会跑 except 分支：
                # 注入的实现可能已经 spawn 了子进程却没人回收，这里显式丢弃。
                await self._discard_client()
                _inc(
                    "mcp_tool_error_total",
                    server=self.config.name,
                    tool="*",
                    reason=MCPTimeoutError.__name__,
                )
                raise MCPTimeoutError(
                    f"MCP server '{self.config.name}' 连接超时（{self.config.timeout_seconds}s）"
                ) from e
            except MCPError:
                raise
            except Exception as e:
                await self._discard_client()
                raise self._map_error(e) from e
        self._connected = True

    async def _discard_client(self) -> None:
        """连接失败后释放可能已拉起的 client / 子进程，并允许重连。

        只处理 adapter 自己创建的 client（注入的 fake 由注入方代管）。
        置空 ``self._client``，让下一次 ``connect()`` 拿到全新 session，
        而不是复用已经被取消过的半开连接。
        """
        self._connected = False
        if self._client is None or not self._owns_client:
            return
        client, self._client = self._client, None
        # 同 close()：CancelledError 是 BaseException，suppress(Exception) 抓不到。
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await client.close()

    async def close(self) -> None:
        """释放 MCP session。只关闭自己创建的 client（注入的 fake 不代管）。"""
        if self._client is not None and self._owns_client:
            try:
                await self._client.close()
            except asyncio.CancelledError:  # pragma: no cover - best effort
                logger.debug("MCP client close 被取消")
            except Exception as e:  # pragma: no cover - best effort
                logger.debug("MCP client close 失败: %s", type(e).__name__)
        self._connected = False

    async def discover_tools(self) -> list[MCPToolSpec]:
        """发现 server 工具，过 allowlist 后归一化 schema。

        - 不在 ``allowed_tools`` 的工具：跳过（计入 ``not_allowed``）；
        - ``inputSchema`` 非法的工具：跳过（计入 ``invalid_schema``）；
        - server 不可用 / 超时：抛对应异常（由调用方决定 fail-closed 策略）。

        **故意不读取 MCP ``annotations``**（``readOnlyHint`` / ``destructiveHint``）：
        那是**远端 server 自述**的安全声明，属于不受信任输入。第三方只要在自家
        代码里加一个 ``readOnlyHint: true`` 就能让本地策略相信它是只读的——正是
        「MCP server 自称 safe 不能覆盖本地 policy」要防的事。因此风险等级只来自
        本地 ``MCPServerConfig.risk_level``（默认 HIGH）。
        ``tests/integration/test_mcp_contract_e2e.py::TestRiskPolicy::
        test_server_annotations_never_lower_local_risk`` 锁住这个行为：任何后续
        「顺手把 annotations 接上」的改动都会让它失败。
        """
        await self.connect()
        client = self._require_client()
        try:
            raw_tools = await client.list_tools()
        except Exception as e:
            mapped = self._map_error(e)
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool="*",
                reason=type(mapped).__name__,
            )
            raise mapped from e

        allowed = set(self.config.allowed_tools)
        specs: list[MCPToolSpec] = []
        for raw in raw_tools or []:
            mcp_name = _read_field(raw, "name")
            if not mcp_name or not isinstance(mcp_name, str):
                continue
            if mcp_name not in allowed:
                # 指标标签用 "*" 而不是 server 提供的原始工具名：被拒绝的工具名
                # 由远端 server 决定（不信任的输入），逐个打成 label 会让
                # Prometheus 时序数随 server 暴露的工具数无界增长。工具名保留在
                # 结构化日志里，排障信息不丢。
                _inc(
                    "mcp_tool_error_total",
                    server=self.config.name,
                    tool="*",
                    reason="not_allowed",
                )
                logger.info("MCP tool 不在 allowlist，跳过: %s/%s", self.config.name, mcp_name)
                continue
            raw_schema = _read_field(raw, "input_schema")
            if raw_schema is _MISSING:
                # server 返回了没有 schema 的工具描述符：无法生成可用的 FC 参数，
                # 静默注册成"无参工具"会让 LLM 凭空造参数。fail closed 跳过。
                _inc(
                    "mcp_tool_error_total",
                    server=self.config.name,
                    tool=mcp_name,
                    reason="missing_schema",
                )
                logger.warning("MCP tool 缺少 inputSchema，跳过 %s/%s", self.config.name, mcp_name)
                continue
            try:
                schema = normalize_input_schema(raw_schema)
            except MCPInvalidSchemaError as e:
                _inc(
                    "mcp_tool_error_total",
                    server=self.config.name,
                    tool=mcp_name,
                    reason="invalid_schema",
                )
                logger.warning(
                    "MCP tool schema 非法，跳过 %s/%s: %s", self.config.name, mcp_name, e
                )
                continue
            description = _read_field(raw, "description")
            # ``_MISSING`` 是"字段不存在"的哨兵，不是描述文本：漏判会让内部哨兵
            # 的 repr（含内存地址）直接进 Function Calling 上下文。
            if description is _MISSING or not description:
                description = ""
            specs.append(
                MCPToolSpec(
                    name=qualified_tool_name(self.config.name, mcp_name),
                    mcp_name=mcp_name,
                    description=str(description),
                    input_schema=schema,
                    risk_level=self.config.risk_level,
                    timeout=self.config.timeout_seconds,
                    server=self.config.name,
                )
            )
        return specs

    async def invoke(self, mcp_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调用一个 MCP 工具。

        执行顺序（每一步都 fail closed）：
        allowlist → payload 字节上限 → 连接 → ``asyncio.wait_for`` 超时 →
        结果归一化 / 错误映射。
        """
        if mcp_name not in set(self.config.allowed_tools):
            # 同 discover_tools：被拒绝的名字不作为指标 label（无界基数），
            # 名字保留在异常消息里给调用方。
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool="*",
                reason="not_allowed",
            )
            raise MCPUnauthorizedToolError(
                f"MCP tool '{mcp_name}' 不在 server '{self.config.name}' 的 allowlist 中"
            )
        args = self._validate_payload(arguments, mcp_name)
        await self.connect()
        client = self._require_client()

        started = time.perf_counter()
        status = "ok"
        try:
            result = await asyncio.wait_for(
                client.call_tool(mcp_name, args), timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError as e:
            status = "timeout"
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool=mcp_name,
                reason=MCPTimeoutError.__name__,
            )
            raise MCPTimeoutError(
                f"MCP tool '{mcp_name}' 调用超时（{self.config.timeout_seconds}s）"
            ) from e
        except MCPError as e:
            status = "error"
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool=mcp_name,
                reason=type(e).__name__,
            )
            raise
        except Exception as e:
            status = "error"
            mapped = self._map_error(e)
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool=mcp_name,
                reason=type(mapped).__name__,
            )
            raise mapped from e
        else:
            normalized = self._normalize_result(result)
            if isinstance(normalized, MCPToolExecutionError):
                status = "tool_error"
                _inc(
                    "mcp_tool_error_total",
                    server=self.config.name,
                    tool=mcp_name,
                    reason=normalized.reason,
                )
                raise normalized
            return normalized
        finally:
            _observe(
                "mcp_tool_duration_seconds",
                time.perf_counter() - started,
                server=self.config.name,
                tool=mcp_name,
            )
            _inc("mcp_tool_call_total", server=self.config.name, tool=mcp_name, status=status)

    # ===== 内部实现 =====

    def _require_client(self) -> MCPClientProtocol:
        if self._client is None:  # pragma: no cover - connect() 之后不可能为 None
            raise MCPUnavailableError(f"MCP server '{self.config.name}' client 未初始化")
        return self._client

    def _validate_payload(self, arguments: dict[str, Any] | None, tool_name: str) -> dict[str, Any]:
        """发送前按 UTF-8 字节数限制 payload 大小。"""
        args = dict(arguments or {})
        try:
            encoded = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool=tool_name,
                reason="payload_not_serializable",
            )
            raise MCPError(f"MCP 调用参数无法序列化为 JSON: {type(e).__name__}") from e
        if len(encoded.encode("utf-8")) > self.config.max_payload_bytes:
            _inc(
                "mcp_tool_error_total",
                server=self.config.name,
                tool=tool_name,
                reason="payload_too_large",
            )
            raise MCPPayloadTooLargeError(
                f"MCP 调用 payload 超过 {self.config.max_payload_bytes} 字节上限"
            )
        return args

    @staticmethod
    def _map_error(exc: Exception) -> MCPError:
        """把 SDK / transport 异常映射到本模块的错误分类。"""
        if isinstance(exc, MCPError):
            return exc
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            if isinstance(exc, TimeoutError):
                return MCPTimeoutError(f"MCP 调用超时: {type(exc).__name__}")
            return MCPUnavailableError(f"MCP server 不可用: {type(exc).__name__}")
        return MCPError(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _normalize_result(result: Any) -> Any:
        """归一化 MCP ``CallToolResult``（兼容 SDK 1.x / 2.x 字段命名）。

        - ``is_error=True`` → 返回 ``MCPToolExecutionError`` 实例（调用方 raise）；
        - ``structured_content`` → 直接返回（dict/JSON 结构化结果）；
        - 否则拼接 text content；都没有则返回 ``None``（由 ``ToolRegistry``
          转成"查询完成，无结果"，避免把 SDK 对象泄进上下文）；
        - 既无 ``content`` 也无 ``is_error`` 的对象（dict / str / fake）
          原样返回。

        ``is_error`` 字段**完全认不出**时（既无 ``isError`` 也无 ``is_error``）
        不能默认当成成功 —— 真实失败会被当成成功返回；也不能沿用
        ``bool(_MISSING) is True`` 顺带当成失败 —— 那样一次成功调用会变成
        失败，还把工具正常输出当成错误文案返回。显式 fail closed 到
        ``unreadable_result``，让指标与日志能指向"结果畸形"而不是"工具失败"。
        """
        if result is None:
            return None
        is_error = _read_field(result, "is_error")
        content = _read_field(result, "content")
        if is_error is _MISSING and content is _MISSING:
            return result
        if is_error is _MISSING:
            return MCPToolExecutionError(
                "MCP 结果既无 isError 也无 is_error 字段，无法判定调用是否成功",
                reason="unreadable_result",
            )
        if bool(is_error):
            text = MCPToolAdapter._extract_text(content)
            return MCPToolExecutionError(text or "MCP tool 返回错误")
        structured = _read_field(result, "structured_content")
        if structured is not _MISSING and structured is not None:
            return structured
        text = MCPToolAdapter._extract_text(content)
        return text if text else None

    @staticmethod
    def _extract_text(content: Any) -> str:
        # ``content`` 字段完全缺失时 ``_read_field`` 返回 ``_MISSING``，而哨兵是
        # truthy 的：直接 ``content or []`` 会去迭代一个 ``object()``，抛
        # TypeError，把"结果畸形"降级成与工具失败无关的通用异常。
        if content is _MISSING or content is None:
            return ""
        parts: list[str] = []
        for item in content or []:
            text = _read_field(item, "text")
            if text is not _MISSING and text is not None:
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)


async def register_mcp_tools(
    registry: Any,
    adapter: MCPToolAdapter,
    *,
    enable_read_cache: bool = False,
    cache_ttl_seconds: int = 300,
) -> list[str]:
    """把 MCP 工具注册进 **现有** ``ToolRegistry``（统一 ToolDefinition）。

    MCP 只是 **tool transport**，不是新的安全边界：注册进来的工具和 native 工具
    共用同一个 ``ToolDefinition``、同一套 ``core.hitl.risk`` 分级、同一道 HITL
    审批闸门、同一个 side-effect ledger。本函数**不建立**任何 MCP 专用权限体系。

    与 native 工具共存的约束：

    - MCP 工具名命名空间化为 ``mcp__{server}__{tool}``，与 native 工具不冲突；
    - 命名冲突（同名工具已存在）时 **跳过而不是覆盖**，绝不顶掉 native 工具；
    - **只注册显式声明为 LOW 的工具**（``risk_level`` 缺失 / 非法 / MEDIUM / HIGH
      一律不注册）。本模块没有写操作的幂等 ledger，不假装有；接入写操作 MCP 工具
      前必须先补幂等 + 风险评审，并让写操作走 T3 durable approval。

    注册进来的只读工具显式声明 ``side_effect=False`` + ``risk_level="low"``：

    - ``side_effect=False``：不进 ``runtime.side_effects`` 幂等 ledger —— 只读
      工具没有外部写副作用可重复，套 ledger 只会给每次读操作增加一次 DB 往返；
    - ``risk_level="low"``（**显式**，而不是留 None）：显式声明的优先级高于
      ``core.hitl.risk.classify_risk`` 的工具名白名单与金额阈值。若留 ``None``，
      一个带 ``amount=999999`` 参数的只读查询会被金额阈值启发式判成 HIGH 并
      触发人工审批 —— 对不产生任何副作用的读操作既无必要又会让审批队列被噪声
      淹没。显式 low 是对「该工具不改任何状态」这一事实的准确声明。

    注意这里的 ``low`` 是**运维显式声明**的结果，不是默认值：只有
    ``MCP_SERVERS`` 里写明 ``"risk_level": "low"`` 的 server 才会走到这一步。
    漏配的后果是「工具不注册」，不是「默认按低风险放行」。

    缓存默认**关闭**（``enable_read_cache=False``）且需要调用方显式开启：
    ``risk_level=low`` 只是本地配置里的声明，不是对第三方 server 返回值的审计
    结论，而 Tool Result Cache 的 scope 只有 ``(user_id, session_id)``。默认不缓存
    是这里 fail closed 的一环；确认某个 server 的返回可安全复用后再逐个开启。
    """
    specs = await adapter.discover_tools()
    existing = set(registry.list_tools())
    registered: list[str] = []
    for spec in specs:
        if spec.risk_level is not RiskLevel.LOW:
            logger.warning(
                "跳过非显式只读的 MCP 工具（只注册 risk_level=low，本模块无写操作"
                "幂等保护）：%s (risk_level=%s)",
                spec.name,
                spec.risk_level.value,
            )
            _inc("mcp_tool_register_total", server=spec.server, outcome="skipped_not_low_risk")
            continue
        if spec.name in existing:
            logger.warning("工具名冲突，跳过 MCP 注册（不覆盖已有工具）: %s", spec.name)
            _inc("mcp_tool_register_total", server=spec.server, outcome="skipped_collision")
            continue

        async def _handler(
            arguments: dict[str, Any],
            _spec: MCPToolSpec = spec,
            _adapter: MCPToolAdapter = adapter,
        ) -> Any:
            return await _adapter.invoke(_spec.mcp_name, arguments)

        registry.register(
            name=spec.name,
            description=spec.description,
            parameters=spec.input_schema,
            handler=_handler,
            cache_policy=ToolCachePolicy(
                enabled=enable_read_cache,
                ttl_seconds=cache_ttl_seconds if enable_read_cache else 0,
            ),
            side_effect=False,
            risk_level=RiskLevel.LOW.value,
            source=SOURCE_MCP,
        )
        existing.add(spec.name)
        registered.append(spec.name)
        _inc("mcp_tool_register_total", server=spec.server, outcome="registered")
    logger.info(
        "MCP server %s 注册 %d 个工具: %s",
        adapter.config.name,
        len(registered),
        registered,
    )
    return registered


class McpSdkClient:
    """官方 ``mcp`` SDK 的最小 client 实现（stdio / sse）。

    SDK 是 **懒加载**依赖：``mcp`` 未安装时抛 ``MCPUnavailableError``，
    不影响 native 工具与整个服务的启动。生命周期由 ``MCPToolAdapter``
    在 ``connect`` / ``close`` 之间管理。

    凭据边界：stdio 子进程只继承 SDK 白名单环境变量（PATH 等），
    **不会**继承服务进程的 API Key；本模块不提供 MCP server 凭据注入。
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._session: Any = None

    def _sdk_read_timeout(self) -> Any:
        """SDK 层的 read timeout 参数（**第二道**防线，第一道是外层 await 超时）。

        SDK 1.x 的 ``ClientSession`` 接受 ``timedelta``，2.x 改为秒（float）。
        这里按实际签名选择类型，避免升级 SDK 后静默失效或直接 TypeError。

        数值上乘 ``_SDK_READ_TIMEOUT_SLACK``，让外层超时确定性地先到期（理由见该常量）。
        """
        import inspect
        from datetime import timedelta

        try:
            from mcp import ClientSession

            param = inspect.signature(ClientSession.__init__).parameters.get("read_timeout_seconds")
        except Exception:  # pragma: no cover - 签名不可读时退回秒
            return float(self.config.timeout_seconds) * _SDK_READ_TIMEOUT_SLACK

        seconds = float(self.config.timeout_seconds) * _SDK_READ_TIMEOUT_SLACK
        if param is not None and "timedelta" in str(param.annotation):
            return timedelta(seconds=seconds)
        return seconds

    async def connect(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession
        except ImportError as e:
            raise MCPUnavailableError(
                "mcp SDK 未安装（外部 MCP 工具不可用；native 工具不受影响）"
            ) from e

        transport = self.config.transport
        if transport == "stdio":
            if not self.config.command:
                raise MCPConfigurationError("stdio MCP server 缺少 command")
            from mcp.client.stdio import StdioServerParameters, stdio_client

            self._transport_cm = stdio_client(
                StdioServerParameters(command=self.config.command, args=list(self.config.args))
            )
        elif transport == "sse":
            if not self.config.url:
                raise MCPConfigurationError("sse MCP server 缺少 url")
            from mcp.client.sse import sse_client

            self._transport_cm = sse_client(
                self.config.url, timeout=min(5.0, self.config.timeout_seconds)
            )
        else:  # pragma: no cover - load_mcp_server_configs 已过滤
            raise MCPConfigurationError(f"不支持的 transport: {transport}")

        try:
            # context manager 的进入/退出刻意**不**包在 wait_for / fail_after 里：
            # SDK 的 stdio 传输在 __aenter__ 时把 anyio cancel scope 绑定到当前
            # task，任何外层包装（wait_for 的临时 task、fail_after 的嵌套 scope）
            # 都会让后续 close() 抛 cancel scope RuntimeError。见
            # ``_await_with_timeout`` 的完整说明。
            streams = await self._transport_cm.__aenter__()
            self._session_cm = ClientSession(
                streams[0],
                streams[1],
                read_timeout_seconds=self._sdk_read_timeout(),
            )
            self._session = await self._session_cm.__aenter__()

            # 握手是唯一需要外层兜底的阶段：它是唯一有「远端往返」的步骤。
            # 进程 spawn 与管道打开由 OS 有界，不额外设限（设限只能靠破坏 cancel
            # scope 的方式做到，见上）。
            import anyio

            try:
                with anyio.fail_after(self.config.timeout_seconds):
                    await self._session.initialize()
            except TimeoutError as e:
                # anyio.fail_after 抛内置 TimeoutError；Python 3.10 上它与
                # asyncio.TimeoutError 不是同一个类（3.11 才合并），显式转换。
                raise asyncio.TimeoutError from e
        except MCPError:
            raise
        except Exception as e:
            await self.close()
            if isinstance(e, asyncio.TimeoutError):
                raise MCPTimeoutError(
                    f"MCP server '{self.config.name}' 连接超时（{self.config.timeout_seconds}s）"
                ) from e
            raise MCPUnavailableError(
                f"MCP server '{self.config.name}' 连接失败: {type(e).__name__}"
            ) from e

    async def list_tools(self) -> list[Any]:
        await self.connect()
        result = await self._session.list_tools()
        return list(getattr(result, "tools", []) or [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await self.connect()
        return await self._session.call_tool(name, arguments or {})

    async def close(self) -> None:
        for cm in (self._session_cm, self._transport_cm):
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    # teardown 期间的取消不是调用方的 bug，cleanup 必须 best-effort
                    # 走完。让它冒出去会把「关一个 MCP session」升级成「整个进程
                    # shutdown 失败」。``CancelledError`` 在 3.8+ 是 BaseException，
                    # 下面那个 ``except Exception`` 抓不到它，必须单独处理。
                    logger.debug("MCP SDK close 被取消")
                except Exception as e:  # pragma: no cover - best effort
                    logger.debug("MCP SDK close 异常: %s", type(e).__name__)
        self._session_cm = None
        self._transport_cm = None
        self._session = None


def build_mcp_adapters(configs: list[MCPServerConfig]) -> list[MCPToolAdapter]:
    """为 allowlist 中 ``enabled`` 的 server 构建适配器（空 allowlist → 空列表）。"""
    return [MCPToolAdapter(cfg) for cfg in configs if cfg.enabled]


__all__ = [
    "FC_NAME_MAX_LEN",
    "MCP_NAME_PREFIX",
    "DEFAULT_RISK_LEVEL",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_NODES",
    "SUPPORTED_PRIMITIVE_TYPES",
    "SUPPORTED_SCHEMA_KEYWORDS",
    "RiskLevel",
    "SOURCE_MCP",
    "SOURCE_NATIVE",
    "MCPConfigurationError",
    "MCPError",
    "MCPInvalidSchemaError",
    "MCPPayloadTooLargeError",
    "MCPServerConfig",
    "MCPToolAdapter",
    "MCPToolExecutionError",
    "MCPToolSpec",
    "MCPTimeoutError",
    "MCPUnauthorizedToolError",
    "MCPUnavailableError",
    "MCPClientProtocol",
    "McpSdkClient",
    "build_mcp_adapters",
    "load_mcp_server_configs",
    "normalize_input_schema",
    "qualified_tool_name",
    "register_mcp_tools",
]
