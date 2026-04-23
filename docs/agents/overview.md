# Agent Graphs Overview

langgraph-kit ships with five built-in agent implementations, ranging from a minimal example to a full-featured coding assistant. All follow the same contract and are registered during application startup.

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
| [Echo](echo-agent.md) | `echo-agent` | Single LLM node + minimal system prompt | Minimal |
| [Basic Deep](basic-deep-agent.md) | `basic-deep-agent` | deepagents framework defaults + minimal system prompt | Low |
| [Reference Deep](reference-deep-agent.md) | `reference-deep-agent` | Full kit feature set | High |
| [Coding](coding-agent.md) | `coding-agent` | Reference + coding overlays | Highest |
| Supervisor | `supervisor-agent` | Routes requests to other agents | Meta |

## Registration

All built-in agents are registered via `register_all()`:

```python
from langgraph_kit.graphs import register_all

async with create_persistence() as (checkpointer, store):
    await register_all(checkpointer, store, mcp_tools=[])
```

This calls each agent's build function and registers the resulting graph with its ID. Agents that fail to build (e.g., missing dependencies) are silently skipped.

## Recursion Limit

> **All deep agents default to `recursion_limit=100`** — significantly higher than LangGraph's native default of `25`, which is not enough for a full-stack deep agent (prompt assembly, middleware, worker round-trips, and tool loops all consume supersteps).

Override per build:

```python
graph, dispatcher = build_reference_deep_agent(
    checkpointer, store, recursion_limit=500
)
```

Override per invocation (wins over the build-time default):

```python
await graph.ainvoke(input_data, config={"recursion_limit": 500})
```

The default lives at `langgraph_kit.graphs.DEFAULT_RECURSION_LIMIT`. Raise it for long autonomous runs; lower it to cap runaway loops in tests or evals.

## Graph Builder

The [Graph Builder](graph-builder.md) package (`core/graph_builder/`) provides shared factories used by the reference and coding agents:
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
