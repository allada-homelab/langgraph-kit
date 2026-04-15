"""MCP tool integration — adapt external protocol capabilities to native model."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk


def adapt_mcp_tool(
    server_name: str,
    name: str,
    description: str,
    fn: Any,
    *,
    risk: ToolRisk = ToolRisk.READ_ONLY,
    tags: list[str] | None = None,
) -> ToolCapability:
    """Wrap an MCP tool function as a ToolCapability."""
    return ToolCapability(
        id=f"mcp_{server_name}_{name}",
        name=name,
        description=description,
        fn=fn,
        tags=[f"mcp:{server_name}", *(tags or [])],
        risk=risk,
        prompt_guidance=(
            f"This tool is provided by the '{server_name}' MCP server. "
            "Treat it like any other tool — use it when it's the best fit."
        ),
    )


def adapt_mcp_tools(
    server_name: str,
    tools: list[dict[str, Any]],
) -> list[ToolCapability]:
    """Adapt a list of MCP tool definitions.

    Each dict should have: name, description, fn, and optionally risk and tags.
    """
    return [
        adapt_mcp_tool(
            server_name,
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            fn=tool_def["fn"],
            risk=ToolRisk(tool_def.get("risk", "read_only")),
            tags=tool_def.get("tags"),
        )
        for tool_def in tools
    ]
