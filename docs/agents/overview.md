# Agent Graphs Overview

langgraph-kit ships with four built-in agent implementations, ranging from a minimal example to a full-featured coding assistant. All follow the same contract and are registered during application startup.

## Agent Contract

Every agent must implement:

```python
def build_graph(checkpointer: Any, store: Any) -> CompiledStateGraph:
    """Build and return a compiled LangGraph graph."""
```

The graph receives a checkpointer (conversation persistence) and a store (key-value persistence) and returns a compiled `StateGraph` ready for execution.

## Built-in Agents

| Agent | ID | Features | Complexity |
|-------|----|----------|------------|
| [Echo](echo-agent.md) | `echo-agent` | Single LLM node | Minimal |
| [Deep](deep-agent.md) | `deep-agent` | deepagents framework | Low |
| [R0](r0-agent.md) | `r0-agent` | Full feature set | High |
| [Coding](coding-agent.md) | `coding-agent` | R0 + coding overlays | Highest |

## Registration

All built-in agents are registered via `register_all()`:

```python
from langgraph_kit.graphs import register_all

async with create_persistence() as (checkpointer, store):
    await register_all(checkpointer, store, mcp_tools=[])
```

This calls each agent's build function and registers the resulting graph with its ID. Agents that fail to build (e.g., missing dependencies) are silently skipped.

## Graph Builder

The [Graph Builder](graph-builder.md) package (`core/graph_builder/`) provides shared factories used by R0 and Coding agents:
- Tool registration helpers (`tools.py`)
- Middleware stack construction (`middleware.py`)
- Command dispatcher setup (`commands.py`)
- Backend factory creation (`backend.py`)

> `graphs/_builder.py` still exists as a backwards-compatible re-export shim.

## Adding a New Agent

1. Create `graphs/my_agent.py` with `build_graph(checkpointer, store)`
2. Register in `graphs/__init__.py` within `register_all()`
3. Optionally use builder utilities for standard features

See the [Quickstart](../getting-started/quickstart.md) for a step-by-step example.
