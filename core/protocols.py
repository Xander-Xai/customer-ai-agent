"""
核心 Protocol 定义（v5.1）
用于替代 Any 类型，提供编译期类型安全。

使用 Protocol（PEP 544）而非 ABC：
- 不需要继承，只需方法签名匹配（鸭子类型）
- mypy strict 模式下可以检查依赖注入的正确性
- 面试时展示对 Python 类型系统的深入理解
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """LLM 客户端协议：所有 LLM 实现必须满足此接口。"""

    async def async_invoke(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 生成回复。"""
        ...

    async def async_invoke_stream(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> Any:
        """流式调用 LLM（SSE）。"""
        ...


@runtime_checkable
class ERPProtocol(Protocol):
    """ERP 适配器协议：所有 ERP 实现必须满足此接口。"""

    async def query_product(self, keyword: str = "") -> list[dict[str, Any]]:
        """查询产品信息。"""
        ...

    async def query_inventory(self, keyword: str = "") -> list[dict[str, Any]]:
        """查询库存信息。"""
        ...

    async def query_order(self, order_id: str = "", customer_id: str = "") -> list[dict[str, Any]]:
        """查询订单信息。"""
        ...

    async def query_customer(self, customer_id: str) -> dict[str, Any] | None:
        """查询客户信息。"""
        ...

    async def get_order_owner(self, order_id: str) -> str | None:
        """P0-03: 最小归属元数据 — 仅返回订单归属 customer_id（不取完整正文）。

        缺失/不存在/查询失败时返回 None（fail closed）。
        """
        ...

    async def resolve_customer_by_user(self, user_id: str | int | None) -> str | None:
        """P0-03: 将可信 authenticated user_id 解析为 ERP customer_id。

        映射必须来自服务端权威数据；缺失时返回 None（fail closed）。
        """
        ...


@runtime_checkable
class KnowledgeBaseProtocol(Protocol):
    """RAG 知识库协议：所有知识库实现必须满足此接口。"""

    async def retrieve(self, request: Any) -> Any:
        """Unified retrieval contract; concrete types live in rag."""
        ...

    async def prefetch_reusable(self, query: str, llm: Any = None) -> dict:
        """Compute reusable query/embedding inputs, never final evidence."""
        ...

    async def query(
        self, text: str, collection: str = "", n_results: int = 3
    ) -> list[dict[str, Any]]:
        """查询知识库，返回相关文档列表。"""
        ...

    async def query_multiple(
        self, text: str, collections: list[str] = ..., n_results: int = 3
    ) -> list[dict[str, Any]]:
        """跨多个 collection 查询并合并结果。"""
        ...

    def get_collection_count(self, collection: str) -> int:
        """获取指定 collection 的文档数。"""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """工具注册中心协议。"""

    def list_tools(self) -> list[str]:
        """列出已注册的工具名。"""
        ...

    def unregister(self, name: str) -> bool:
        """撤销注册（回滚用），返回是否真的移除过一个工具。

        MCP 工具是运行时叠加进同一个注册表的，多 server 初始化可能部分成功后失败；
        没有撤销能力就无法把注册表恢复到「本次 attempt 之前」的状态。
        """
        ...

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """获取供 LLM Function Calling 使用的工具定义。"""
        ...

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], tool_call_id: str | None = None
    ) -> Any:
        """执行指定工具。"""
        ...

    async def execute_raw(
        self, tool_name: str, arguments: dict[str, Any], tool_call_id: str | None = None
    ) -> Any:
        """执行工具并保留结构化结果（可选的 Context Engineering 路径）。"""
        ...


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """会话管理器协议。"""

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话。"""
        ...

    async def add_message(self, session_id: str, message: str, role: str = "user") -> None:
        """添加消息到会话。"""
        ...

    async def get_context_messages(
        self, session_id: str, max_messages: int = 6
    ) -> list[dict[str, str]]:
        """获取上下文消息列表。"""
        ...
