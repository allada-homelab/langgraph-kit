"""Coverage fill — MCP server factory + mount with a fake FastMCP.

MCP server machinery is exercised here without a live MCP client by
injecting a fake ``FastMCP`` class via ``sys.modules``. We verify:

- One MCP tool is registered per kit agent in the registry.
- Each tool's LLM-visible name is ``invoke_<agent_id>``.
- Invoking a tool drives the agent graph via ``ainvoke``.
- ``mount_mcp_server`` calls ``app.mount`` at the expected path.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from langgraph_kit import registry as registry_mod
from langgraph_kit.registry import AgentMetadata, register


class _FakeMCP:
    """Minimal stand-in for ``mcp.server.fastmcp.FastMCP``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.registered: list[dict[str, Any]] = []
        self._streamable = MagicMock()

    def tool(self, *, name: str, description: str) -> Any:
        def _decorator(fn: Any) -> Any:
            self.registered.append({"name": name, "description": description, "fn": fn})
            return fn

        return _decorator

    def streamable_http_app(self) -> Any:
        return self._streamable


@pytest.fixture
def _fake_mcp_module() -> Iterator[None]:
    """Inject fake ``mcp.server.fastmcp`` module into sys.modules for the test."""
    fake = MagicMock(FastMCP=_FakeMCP)
    originals: dict[str, Any] = {}
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        originals[name] = sys.modules.get(name)
    sys.modules["mcp"] = MagicMock()
    sys.modules["mcp.server"] = MagicMock()
    sys.modules["mcp.server.fastmcp"] = fake
    try:
        yield
    finally:
        for name, orig in originals.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


@pytest.fixture
def _clean_registry() -> Iterator[None]:
    snap_registry = dict(registry_mod._registry)
    snap_dispatchers = dict(registry_mod._dispatchers)
    snap_meta = dict(registry_mod._metadata)
    registry_mod._registry.clear()
    registry_mod._dispatchers.clear()
    registry_mod._metadata.clear()
    try:
        yield
    finally:
        registry_mod._registry.clear()
        registry_mod._dispatchers.clear()
        registry_mod._metadata.clear()
        registry_mod._registry.update(snap_registry)
        registry_mod._dispatchers.update(snap_dispatchers)
        registry_mod._metadata.update(snap_meta)


class _FakeGraph:
    def __init__(self, response: str) -> None:
        self._response = response

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        _ = input_data
        _ = config

        class _M:
            content = self._response

        return {"messages": [_M()]}


@pytest.mark.usefixtures("_fake_mcp_module", "_clean_registry")
def test_create_mcp_server_registers_one_tool_per_agent() -> None:
    from langgraph_kit.contrib.mcp_server import create_mcp_server

    register(
        "alpha",
        _FakeGraph("alpha-reply"),
        metadata=AgentMetadata(description="alpha agent"),
    )
    register(
        "beta-agent",
        _FakeGraph("beta-reply"),
        metadata=AgentMetadata(description="beta agent"),
    )

    mcp = create_mcp_server("test-server")

    assert isinstance(mcp, _FakeMCP)
    assert mcp.name == "test-server"
    registered_names = [r["name"] for r in mcp.registered]
    # Dashes replaced with underscores in the tool name.
    assert registered_names == ["invoke_alpha", "invoke_beta_agent"]


@pytest.mark.usefixtures("_fake_mcp_module", "_clean_registry")
@pytest.mark.asyncio
async def test_registered_tool_invokes_graph_and_returns_response() -> None:
    from langgraph_kit.contrib.mcp_server import create_mcp_server

    register(
        "echo",
        _FakeGraph("echo-content"),
        metadata=AgentMetadata(),
    )

    mcp = create_mcp_server()
    tool_fn = mcp.registered[0]["fn"]
    assert await tool_fn("hello") == "echo-content"


@pytest.mark.usefixtures("_fake_mcp_module", "_clean_registry")
@pytest.mark.asyncio
async def test_registered_tool_surfaces_graph_errors_as_strings() -> None:
    from langgraph_kit.contrib.mcp_server import create_mcp_server

    class _ErrGraph:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            _ = args
            _ = kwargs
            msg = "graph exploded"
            raise RuntimeError(msg)

    register("boom", _ErrGraph(), metadata=AgentMetadata())
    mcp = create_mcp_server()
    tool_fn = mcp.registered[0]["fn"]
    result = await tool_fn("hi")
    assert "Error invoking boom" in result
    assert "graph exploded" in result


@pytest.mark.usefixtures("_fake_mcp_module")
def test_mount_mcp_server_calls_app_mount_at_path() -> None:
    from langgraph_kit.contrib.mcp_server import mount_mcp_server

    app = MagicMock()
    mcp = _FakeMCP("x")

    mount_mcp_server(app, mcp, path="/mymcp")
    app.mount.assert_called_once()
    args, kwargs = app.mount.call_args
    assert args[0] == "/mymcp" or kwargs.get("path") == "/mymcp"
