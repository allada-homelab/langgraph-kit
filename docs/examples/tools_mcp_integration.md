# Tools — MCP integration

Shows the `AgentConfig.mcp_servers` JSON shape the kit consumes at
startup, plus how MCP tools land as native `ToolCapability` objects in
the agent's registry. The demo introspects the tool catalogue declared
by [`examples/_sample_mcp_server.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/_sample_mcp_server.py)
without booting a real subprocess.

```bash
uv run python -m examples.tools_mcp_integration
```

```python
--8<-- "examples/tools_mcp_integration.py"
```
