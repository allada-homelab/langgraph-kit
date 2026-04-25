"""Sample stdio MCP server used by ``examples/tools_mcp_integration.py``.

A real MCP integration takes the JSON config from
``AgentConfig.mcp_servers`` and starts each child server as a stdio
subprocess; the kit's MCP adapter wraps every exposed tool as a native
:class:`ToolCapability` registered into the agent's tool list.

This file is named with a leading underscore so the
:mod:`examples.run_all` smoke driver doesn't try to execute it as a
standalone demo. The companion example imports it and reads its tool
declaration without needing an actual MCP runtime, so the demo stays
hermetic.
"""

from __future__ import annotations

# Public-shape declaration the demo introspects. Mirrors what an MCP
# server would advertise via ``ListTools`` over stdio. Real servers
# return ``Tool`` objects from the official ``mcp`` package; we keep
# the schema shape only, so the demo doesn't need a server runtime.
SAMPLE_TOOLS: list[dict[str, str]] = [
    {
        "name": "weather_lookup",
        "description": "Return the current temperature for a city.",
    },
    {
        "name": "weather_forecast",
        "description": "Return the 3-day forecast for a city.",
    },
]
