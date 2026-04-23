"""Coverage fill — ``MCPClientManager`` config parsing + transport dispatch.

The manager is exercised here without real MCP servers by monkey-
patching the transport client constructors. We drive every branch of
``connect_all`` / ``_connect_and_load`` / ``close`` so:

- Empty / invalid ``MCP_SERVERS`` JSON is tolerated.
- Unknown transports are logged and skipped.
- stdio transport path loads tools via ``load_mcp_tools``.
- Errors during connect are caught per-server.
- ``close`` tears down sessions and contexts in reverse order.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from langgraph_kit.core.plugins.mcp_client import MCPClientManager


class _FakeCtx:
    """Async context manager returning (read, write) on ``__aenter__``."""

    def __init__(self, closed_calls: list[str]) -> None:
        self.closed_calls = closed_calls

    async def __aenter__(self) -> tuple[Any, Any]:
        return ("read", "write")

    async def __aexit__(self, *exc_info: Any) -> None:
        _ = exc_info
        self.closed_calls.append("ctx-closed")


class _FakeHttpCtx(_FakeCtx):
    async def __aenter__(self) -> tuple[Any, Any, Any]:  # type: ignore[override]
        return ("read", "write", "extra")


class _FakeSession:
    def __init__(
        self, tools: list[Any], closed_calls: list[str], *, raise_init: bool = False
    ) -> None:
        self._tools = tools
        self._closed_calls = closed_calls
        self._raise_init = raise_init

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        _ = exc_info
        self._closed_calls.append("session-closed")

    async def initialize(self) -> None:
        if self._raise_init:
            msg = "initialize failed"
            raise RuntimeError(msg)


@pytest.fixture
def fake_mcp_modules() -> Iterator[dict[str, Any]]:
    """Install stub MCP + langchain_mcp_adapters modules in sys.modules.

    Yields a mutable dict that tests can introspect or tweak per-case.
    """
    state: dict[str, Any] = {
        "loaded_tools": [],
        "stdio_ctx": None,
        "session": None,
        "closed_calls": [],
    }

    fake_mcp = MagicMock()
    fake_mcp.ClientSession = MagicMock(
        side_effect=lambda *args, **kwargs: state["session"]
    )
    fake_mcp.StdioServerParameters = MagicMock()

    stdio_client_mock = MagicMock(side_effect=lambda params: state["stdio_ctx"])
    sse_client_mock = MagicMock(side_effect=lambda **kwargs: state["stdio_ctx"])
    streamable_client_mock = MagicMock(
        side_effect=lambda **kwargs: state["stdio_ctx"]
    )

    adapters_stub = MagicMock(
        load_mcp_tools=AsyncMock(side_effect=lambda *a, **kw: state["loaded_tools"])
    )

    fakes: dict[str, Any] = {
        "mcp": fake_mcp,
        "mcp.client": MagicMock(),
        "mcp.client.stdio": MagicMock(stdio_client=stdio_client_mock),
        "mcp.client.sse": MagicMock(sse_client=sse_client_mock),
        "mcp.client.streamable_http": MagicMock(
            streamable_http_client=streamable_client_mock
        ),
        "langchain_mcp_adapters": MagicMock(),
        "langchain_mcp_adapters.tools": adapters_stub,
    }

    originals = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        yield state
    finally:
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_invalid_json_is_tolerated() -> None:
    mgr = MCPClientManager("{not valid json")
    # No configs loaded — connect_all would return an empty contribution.
    assert mgr._configs == []


def test_empty_string_yields_no_configs() -> None:
    mgr = MCPClientManager("")
    assert mgr._configs == []


@pytest.mark.asyncio
async def test_connect_all_with_no_configs_returns_empty_contribution() -> None:
    mgr = MCPClientManager("")
    contribution = await mgr.connect_all()
    assert contribution.plugin_id == "mcp_servers"
    assert contribution.tools == []


@pytest.mark.asyncio
async def test_unknown_transport_is_skipped(
    fake_mcp_modules: dict[str, Any],
) -> None:
    servers_json = '[{"name":"odd","transport":"unknown"}]'
    mgr = MCPClientManager(servers_json)
    contribution = await mgr.connect_all()
    # Unknown transport branch returns [] so no tools get wrapped.
    assert contribution.tools == []


@pytest.mark.asyncio
async def test_stdio_transport_loads_tools_and_wraps_them(
    fake_mcp_modules: dict[str, Any],
) -> None:
    closed_calls: list[str] = []
    fake_mcp_modules["closed_calls"] = closed_calls
    fake_mcp_modules["stdio_ctx"] = _FakeCtx(closed_calls)
    tools = [MagicMock(description="tool-one"), MagicMock(description="tool-two")]
    tools[0].name = "t1"  # explicit string (MagicMock auto-attr is a Mock)
    tools[1].name = "t2"
    fake_mcp_modules["session"] = _FakeSession(
        tools=tools,
        closed_calls=closed_calls,
    )
    fake_mcp_modules["loaded_tools"] = fake_mcp_modules["session"]._tools

    servers_json = (
        '[{"name":"my-server","transport":"stdio",'
        '"command":"echo","args":["hello"]}]'
    )
    mgr = MCPClientManager(servers_json)
    contribution = await mgr.connect_all()

    # Each tool became a ToolCapability on the contribution.
    assert len(contribution.tools) == 2

    # close() unwinds sessions and contexts.
    await mgr.close()
    assert "session-closed" in closed_calls
    assert "ctx-closed" in closed_calls


@pytest.mark.asyncio
async def test_sse_transport_dispatches_to_sse_client(
    fake_mcp_modules: dict[str, Any],
) -> None:
    closed_calls: list[str] = []
    fake_mcp_modules["stdio_ctx"] = _FakeCtx(closed_calls)
    sse_tool = MagicMock(description="")
    sse_tool.name = "sse-tool"
    fake_mcp_modules["session"] = _FakeSession(
        tools=[sse_tool],
        closed_calls=closed_calls,
    )
    fake_mcp_modules["loaded_tools"] = fake_mcp_modules["session"]._tools

    servers_json = '[{"name":"web","transport":"sse","url":"http://x"}]'
    mgr = MCPClientManager(servers_json)
    contribution = await mgr.connect_all()
    assert len(contribution.tools) == 1


@pytest.mark.asyncio
async def test_http_transport_unpacks_three_tuple(
    fake_mcp_modules: dict[str, Any],
) -> None:
    closed_calls: list[str] = []
    fake_mcp_modules["stdio_ctx"] = _FakeHttpCtx(closed_calls)
    http_tool = MagicMock(description="")
    http_tool.name = "http-tool"
    fake_mcp_modules["session"] = _FakeSession(
        tools=[http_tool],
        closed_calls=closed_calls,
    )
    fake_mcp_modules["loaded_tools"] = fake_mcp_modules["session"]._tools

    servers_json = '[{"name":"api","transport":"http","url":"http://x"}]'
    mgr = MCPClientManager(servers_json)
    contribution = await mgr.connect_all()
    assert len(contribution.tools) == 1


@pytest.mark.asyncio
async def test_server_connect_error_is_caught_per_server(
    fake_mcp_modules: dict[str, Any],
) -> None:
    closed_calls: list[str] = []
    fake_mcp_modules["stdio_ctx"] = _FakeCtx(closed_calls)
    fake_mcp_modules["session"] = _FakeSession(
        tools=[],
        closed_calls=closed_calls,
        raise_init=True,  # initialize() raises → caught by outer try/except
    )

    servers_json = (
        '[{"name":"broken","transport":"stdio","command":"nope","args":[]},'
        '{"name":"other","transport":"unknown"}]'
    )
    mgr = MCPClientManager(servers_json)
    # Must not raise even though the first server's initialize raises.
    contribution = await mgr.connect_all()
    assert contribution.plugin_id == "mcp_servers"


@pytest.mark.asyncio
async def test_close_swallows_errors_during_teardown(
    fake_mcp_modules: dict[str, Any],
) -> None:
    class _Raising:
        async def __aexit__(self, *exc: Any) -> None:
            _ = exc
            msg = "cleanup failed"
            raise RuntimeError(msg)

    mgr = MCPClientManager("")
    # Pre-populate sessions/contexts so close() has something to iterate.
    mgr._sessions = [_Raising()]  # type: ignore[assignment]
    mgr._contexts = [_Raising()]  # type: ignore[assignment]
    # Must not raise — errors during teardown are logged and swallowed.
    await mgr.close()
    assert mgr._sessions == []
    assert mgr._contexts == []
