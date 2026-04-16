"""MCP client manager — connects to MCP servers and adapts tools for the agent system."""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph_kit.core.plugins.mcp import adapt_mcp_tool
from langgraph_kit.core.plugins.registry import PluginContribution
from langgraph_kit.core.tools.capability import ToolCapability

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to MCP servers and produces ToolCapability objects.

    Parses server configs from a JSON string, connects via
    ``langchain-mcp-adapters``, and adapts discovered tools through
    :class:`MCPToolAdapter`.

    Usage::

        mgr = MCPClientManager(mcp_servers_json)
        contribution = await mgr.connect_all()
        # contribution.tools is a list of ToolCapability objects
        await mgr.close()
    """

    def __init__(self, servers_json: str) -> None:
        super().__init__()
        self._configs: list[dict[str, Any]] = []
        self._sessions: list[Any] = []
        self._contexts: list[Any] = []

        if servers_json.strip():
            try:
                self._configs = json.loads(servers_json)
            except json.JSONDecodeError:
                logger.warning("Invalid MCP_SERVERS JSON — no MCP servers will load")

    async def connect_all(self) -> PluginContribution:
        """Connect to all configured MCP servers and return adapted tools."""
        if not self._configs:
            return PluginContribution(plugin_id="mcp_servers")

        all_capabilities: list[ToolCapability] = []

        for config in self._configs:
            server_name = config.get("name", "unnamed")
            transport = config.get("transport", "stdio")

            try:
                tools = await self._connect_and_load(config, transport, server_name)
                for tool in tools:
                    cap = adapt_mcp_tool(
                        server_name,
                        name=tool.name,
                        description=tool.description or "",
                        fn=tool,
                    )
                    all_capabilities.append(cap)
                logger.info(
                    "MCP server '%s' (%s): loaded %d tools",
                    server_name,
                    transport,
                    len(tools),
                )
            except Exception:
                logger.warning(
                    "MCP server '%s' failed to connect — skipping",
                    server_name,
                    exc_info=True,
                )

        return PluginContribution(plugin_id="mcp_servers", tools=all_capabilities)

    async def _connect_and_load(
        self,
        config: dict[str, Any],
        transport: str,
        server_name: str,
    ) -> list[Any]:
        """Connect to a single MCP server and return LangChain tools."""

        if transport == "stdio":
            return await self._connect_stdio(config, server_name)
        if transport == "sse":
            return await self._connect_sse(config, server_name)
        if transport in ("http", "streamablehttp"):
            return await self._connect_http(config, server_name)

        logger.warning(
            "Unknown MCP transport '%s' for server '%s'", transport, server_name
        )
        return []

    async def _connect_stdio(
        self, config: dict[str, Any], server_name: str
    ) -> list[Any]:
        from langchain_mcp_adapters.tools import (
            load_mcp_tools,  # pyright: ignore[reportMissingModuleSource]
        )
        from mcp import (  # pyright: ignore[reportMissingModuleSource]
            ClientSession,
            StdioServerParameters,
        )
        from mcp.client.stdio import (
            stdio_client,  # pyright: ignore[reportMissingModuleSource]
        )

        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        self._contexts.append(ctx)

        session = ClientSession(read, write)
        await session.__aenter__()
        self._sessions.append(session)

        await session.initialize()
        return await load_mcp_tools(session, server_name=server_name)

    async def _connect_sse(self, config: dict[str, Any], server_name: str) -> list[Any]:
        from langchain_mcp_adapters.tools import (
            load_mcp_tools,  # pyright: ignore[reportMissingModuleSource]
        )
        from mcp import ClientSession  # pyright: ignore[reportMissingModuleSource]
        from mcp.client.sse import (
            sse_client,  # pyright: ignore[reportMissingModuleSource]
        )

        ctx = sse_client(
            url=config["url"],
            headers=config.get("headers", {}),
        )
        read, write = await ctx.__aenter__()
        self._contexts.append(ctx)

        session = ClientSession(read, write)
        await session.__aenter__()
        self._sessions.append(session)

        await session.initialize()
        return await load_mcp_tools(session, server_name=server_name)

    async def _connect_http(
        self, config: dict[str, Any], server_name: str
    ) -> list[Any]:
        from langchain_mcp_adapters.tools import (
            load_mcp_tools,  # pyright: ignore[reportMissingModuleSource]
        )
        from mcp import ClientSession  # pyright: ignore[reportMissingModuleSource]
        from mcp.client.streamable_http import (
            streamable_http_client,  # pyright: ignore[reportMissingModuleSource]
        )

        ctx: Any = streamable_http_client(  # pyright: ignore[reportUnknownVariableType]
            url=config["url"],
            headers=config.get("headers", {}),  # pyright: ignore[reportCallIssue]
        )
        read: Any
        write: Any
        read, write, _ = await ctx.__aenter__()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        self._contexts.append(ctx)

        session = ClientSession(read, write)  # pyright: ignore[reportUnknownArgumentType]
        await session.__aenter__()
        self._sessions.append(session)

        await session.initialize()
        return await load_mcp_tools(session, server_name=server_name)

    async def close(self) -> None:
        """Close all MCP sessions and transport contexts."""
        for session in reversed(self._sessions):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing MCP session", exc_info=True)
        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing MCP transport", exc_info=True)
        self._sessions.clear()
        self._contexts.clear()
