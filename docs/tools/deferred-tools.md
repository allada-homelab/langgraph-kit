# Deferred Tools

**Source:** `src/langgraph_kit/core/tools/deferred.py`

Deferred tools enable lazy tool discovery — tools are registered by name but their full schema is only loaded when the agent actually needs them. This reduces the initial tool surface while keeping capabilities available on demand.

## Class: DeferredToolRegistry

Manages tools that are registered by metadata but not fully loaded until requested.

### Methods

| Method | Description |
|--------|-------------|
| `register(name, loader_fn)` | Register a deferred tool with a lazy loader |
| `resolve(name)` | Load and return the full tool capability |
| `list_available()` | List registered deferred tool names |

## Use Case

When an agent has access to many specialized tools but only needs a few per conversation, deferred registration avoids overwhelming the LLM's tool selection with irrelevant options. The agent can use a "search tools" tool to discover what's available and request specific tools to be loaded.

## Integration

The deferred tool registry works with the `register_search_tool()` builder function, which gives the agent a tool to search for and load deferred capabilities at runtime.
