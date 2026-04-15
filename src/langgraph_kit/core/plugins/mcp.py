"""MCP resource and tool integration — adapt external protocol capabilities to native model."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapts MCP-provided tools into the agent's ToolCapability model.

    MCP tools come as dictionaries with name, description, and input_schema.
    This adapter wraps them as ToolCapability objects that can be registered
    in the ToolRegistry like any native tool.
    """

    def __init__(self, server_name: str) -> None:
        super().__init__()
        self._server = server_name

    def adapt_tool(
        self,
        name: str,
        description: str,
        fn: Any,
        *,
        risk: ToolRisk = ToolRisk.READ_ONLY,
        tags: list[str] | None = None,
    ) -> ToolCapability:
        """Wrap an MCP tool function as a ToolCapability."""
        return ToolCapability(
            id=f"mcp_{self._server}_{name}",
            name=name,
            description=description,
            fn=fn,
            tags=[f"mcp:{self._server}", *(tags or [])],
            risk=risk,
            prompt_guidance=(
                f"This tool is provided by the '{self._server}' MCP server. "
                "Treat it like any other tool — use it when it's the best fit."
            ),
        )

    def adapt_many(
        self,
        tools: list[dict[str, Any]],
    ) -> list[ToolCapability]:
        """Adapt a list of MCP tool definitions.

        Each dict should have: name, description, fn, and optionally risk and tags.
        """
        capabilities: list[ToolCapability] = []
        for tool_def in tools:
            cap = self.adapt_tool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                fn=tool_def["fn"],
                risk=ToolRisk(tool_def.get("risk", "read_only")),
                tags=tool_def.get("tags"),
            )
            capabilities.append(cap)
        return capabilities


