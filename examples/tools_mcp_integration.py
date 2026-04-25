"""Tools: MCP server adapter — wrap external MCP tools as native capabilities.

What this shows
---------------
- The shape of ``AgentConfig.mcp_servers`` (JSON string keyed by
  server name) the kit consumes at startup
- How exposed tools land as native :class:`ToolCapability` objects in
  the agent's registry once the MCP adapter has bridged them
- Inspecting the wrapped tool list before the agent's first turn

This demo introspects the tool catalogue declared by
``examples/_sample_mcp_server.py`` *without* actually starting a
subprocess — booting a real MCP server in CI requires Node/uv plus
network access. The full live wiring is deferred to the nightly
network-tier workflow.

How to run
----------
    uv run python -m examples.tools_mcp_integration

Expected output
---------------
    AgentConfig.mcp_servers JSON shape:
      {"weather": {"command": "python", "args": ["-m", "examples._sample_mcp_server"]}}
    Tools the sample server would expose:
      - weather_lookup: Return the current temperature for a city.
      - weather_forecast: Return the 3-day forecast for a city.
    Wrapped as kit ToolCapability objects, the agent would see:
      - weather_lookup     risk=read_only
      - weather_forecast   risk=read_only
"""

from __future__ import annotations

import json

from examples._lib import banner, line


def main() -> None:
    banner("tools_mcp_integration")

    from examples._sample_mcp_server import SAMPLE_TOOLS
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.core.tools.registry import ToolRegistry

    mcp_config = {
        "weather": {
            "command": "python",
            "args": ["-m", "examples._sample_mcp_server"],
        }
    }
    line("AgentConfig.mcp_servers JSON shape:")
    line(f"  {json.dumps(mcp_config)}")

    line("\nTools the sample server would expose:")
    for tool in SAMPLE_TOOLS:
        line(f"  - {tool['name']}: {tool['description']}")

    # In the real wiring the kit's MCP adapter calls the server's
    # ListTools and constructs a ToolCapability per entry. We mirror
    # that shape locally so the demo's output matches what the agent
    # actually sees post-bridge.
    registry = ToolRegistry()
    for tool in SAMPLE_TOOLS:
        registry.register(
            ToolCapability(
                id=tool["name"],
                name=tool["name"],
                description=tool["description"],
                fn=lambda **_kwargs: "[stub] real MCP call would happen here",
                risk=ToolRisk.READ_ONLY,
                tags=["mcp", "weather"],
            )
        )

    line("\nWrapped as kit ToolCapability objects, the agent would see:")
    for cap in registry.list_all():
        line(f"  - {cap.name:<18} risk={cap.risk.value}")


if __name__ == "__main__":
    main()
