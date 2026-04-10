# MCP Adapter

**Source:** `src/langgraph_kit/core/plugins/mcp.py`

Adapts MCP (Model Context Protocol) tools and resources into langgraph-kit's native model.

## MCPToolAdapter

Wraps MCP tools as `ToolCapability` instances.

### Methods

#### adapt_tool(mcp_tool) -> ToolCapability

Wrap a single MCP tool into a `ToolCapability`:
- ID derived from the MCP tool name
- Description from the MCP tool description
- Risk defaults to `READ_ONLY` (can be overridden)
- The callable wraps the MCP tool invocation

#### adapt_many(mcp_tools) -> list[ToolCapability]

Adapt a list of MCP tools in bulk.

## MCPResourceReader

Reads MCP resources and formats them for agent consumption.

### Methods

| Method | Description |
|--------|-------------|
| `register_resource(uri, name, description, read_fn)` | Register an MCP resource |
| `list_resources()` | List available resources |
| `read_resource(uri)` | Read and return resource content |

## MCPClientManager

**Source:** `src/langgraph_kit/core/plugins/mcp_client.py`

Manages connections to MCP servers.

### Methods

| Method | Description |
|--------|-------------|
| `connect(server_config)` | Connect to an MCP server |
| `disconnect(name)` | Disconnect from a server |
| `get_tools(name)` | Get tools from a connected server |
| `get_resources(name)` | Get resources from a connected server |

## Usage

```python
from langgraph_kit.core.plugins.mcp import MCPToolAdapter

adapter = MCPToolAdapter()

# Adapt MCP tools for the tool registry
mcp_capabilities = adapter.adapt_many(mcp_tools_from_server)
for cap in mcp_capabilities:
    tool_registry.register(cap)
```
