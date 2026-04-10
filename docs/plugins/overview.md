# Plugins & MCP Overview

The plugin system extends agent capabilities through two mechanisms: MCP (Model Context Protocol) server integration and Python plugin files.

## Components

| Module | Purpose |
|--------|---------|
| [MCP Adapter](mcp-adapter.md) | Wrapping MCP tools as native `ToolCapability` |
| [Plugin Loader](plugin-loader.md) | Python plugin files with `contribute()` functions |

## MCP Integration

MCP (Model Context Protocol) allows agents to connect to external tool servers that provide additional capabilities. langgraph-kit wraps MCP tools into the native `ToolCapability` model so they integrate seamlessly with the tool registry, risk filtering, and prompt assembly.

```
MCP Server (external process)
    │
    ▼
MCPClientManager (manages connections)
    │
    ▼
MCPToolAdapter (wraps as ToolCapability)
    │
    ▼
ToolRegistry (standard registration)
    │
    ▼
Agent can use MCP tools like any other tool
```

## Python Plugins

For extensions that don't need a separate server process, Python plugin files provide a simpler mechanism:

```python
# plugins/my_plugin.py
def contribute():
    """Called during agent startup."""
    return PluginContribution(
        tools=[my_custom_tool],
        sections=[my_prompt_section],
    )
```

## Configuration

MCP servers are configured via the `mcp_servers` field in `AgentConfig` as a JSON string:

```python
AgentConfig(
    mcp_servers='[{"name": "filesystem", "command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "/path"]}]'
)
```

Python plugins are loaded from the `plugins_dir` directory:

```python
AgentConfig(plugins_dir="/path/to/plugins/")
```
