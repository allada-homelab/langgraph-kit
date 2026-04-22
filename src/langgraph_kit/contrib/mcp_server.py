"""MCP server mode — expose registered agents as MCP tools.

Allows external MCP clients (Claude Desktop, Cursor, etc.) to invoke
langgraph-kit agents as tools.

Usage::

    from langgraph_kit.contrib.mcp_server import create_mcp_server, mount_mcp_server

    mcp = create_mcp_server()
    mount_mcp_server(app, mcp, path="/mcp")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_mcp_server(name: str = "langgraph-kit") -> Any:
    """Create a FastMCP server with one tool per registered agent.

    Each agent becomes an MCP tool named ``invoke_{agent_id}`` that accepts
    a message string and optional thread_id, returning the agent's response.

    Returns the ``FastMCP`` instance.
    """
    from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingModuleSource]

    from langgraph_kit.registry import get_all, get_metadata

    mcp = FastMCP(name)
    agents = get_all()

    for agent_id, graph in agents.items():
        meta = get_metadata(agent_id)
        description = meta.description or f"Invoke the {agent_id} agent with a message"

        # Create a closure to capture agent_id and graph
        _register_agent_tool(mcp, agent_id, graph, description)

    logger.info("MCP server created with %d agent tools", len(agents))
    return mcp


def _register_agent_tool(
    mcp: Any,
    agent_id: str,
    graph: Any,
    description: str,
) -> None:
    """Register a single agent as an MCP tool."""
    tool_name = f"invoke_{agent_id.replace('-', '_')}"

    @mcp.tool(name=tool_name, description=description)
    async def invoke_agent(message: str, thread_id: str = "") -> str:
        from uuid import uuid4

        from langchain_core.messages import (
            HumanMessage,  # pyright: ignore[reportMissingModuleSource]
        )

        tid = thread_id or str(uuid4())
        config: dict[str, Any] = {"configurable": {"thread_id": tid}}
        input_data = {"messages": [HumanMessage(content=message)]}

        try:
            result = await graph.ainvoke(input_data, config=config)
            last = result["messages"][-1]
            return last.content if hasattr(last, "content") else str(last)
        except Exception as exc:
            return f"Error invoking {agent_id}: {exc}"


def mount_mcp_server(app: Any, mcp: Any, path: str = "/mcp") -> None:
    """Mount an MCP server as a sub-application on a FastAPI app.

    Uses ``mcp.streamable_http_app()`` to create a Starlette ASGI app
    and mounts it at the given path.

    Parameters
    ----------
    app:
        The FastAPI application.
    mcp:
        The ``FastMCP`` instance from ``create_mcp_server()``.
    path:
        URL path to mount the MCP server at.
    """
    starlette_app = mcp.streamable_http_app()
    app.mount(path, starlette_app)
    logger.info("MCP server mounted at %s", path)
