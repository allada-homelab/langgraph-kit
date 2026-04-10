# Quickstart

This guide walks through building and running your first langgraph-kit agent.

## 1. Install the Package

```bash
uv add "langgraph-kit[fastapi] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"
```

## 2. Configure at Startup

```python
from langgraph_kit import AgentConfig, configure

configure(AgentConfig(
    llm_model="gpt-4o-mini",
    llm_api_key="sk-...",
    database_url="sqlite:///checkpoints.db",
))
```

## 3. Build a Minimal Agent

The simplest agent uses the echo agent pattern — a single LLM node in a LangGraph StateGraph:

```python
import uuid
from langgraph_kit import build_llm, create_persistence, register, stream_agent_events

# The echo agent is built-in
from langgraph_kit.graphs.echo_agent import build_graph


async def main():
    async with create_persistence() as (checkpointer, store):
        # Build and register the graph
        graph = build_graph(checkpointer, store)
        register("my-agent", graph)

        # Run a conversation
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_data = {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        }

        async for chunk in stream_agent_events(graph, input_data, config):
            print(chunk, end="")
```

## 4. Register All Built-in Agents

To register every built-in agent (echo, deep, r0, coding):

```python
from langgraph_kit.graphs import register_all

async with create_persistence() as (checkpointer, store):
    await register_all(checkpointer, store, mcp_tools=[])
```

## 5. Expose via FastAPI

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from langgraph_kit import AgentConfig, configure, create_persistence
from langgraph_kit.contrib.fastapi import create_agent_router
from langgraph_kit.graphs import register_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure(AgentConfig(
        llm_model="gpt-4o-mini",
        llm_api_key="sk-...",
    ))
    async with create_persistence() as (checkpointer, store):
        await register_all(checkpointer, store, mcp_tools=[])
        app.state.store = store
        yield


app = FastAPI(lifespan=lifespan)

# get_current_user is your auth dependency
agent_router = create_agent_router(get_current_user=get_current_user)
app.include_router(agent_router, prefix="/api/v1")
```

This gives you endpoints for:
- `GET /api/v1/agents/` — list agents
- `POST /api/v1/agents/{id}/stream` — stream tokens (SSE)
- `POST /api/v1/agents/{id}/invoke` — full response (JSON)
- And [many more](../integrations/fastapi.md)

## 6. Create a Custom Agent

Use the CLI to scaffold a new agent:

```bash
uv run python -m langgraph_kit.cli new my-custom-agent --output-dir ./agents/
```

Or create one manually following the [agent contract](../agents/overview.md):

```python
# my_agent.py
from typing import Any
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph_kit import build_llm


def build_graph(checkpointer: Any, store: Any) -> Any:
    llm = build_llm()

    async def agent_node(state: MessagesState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=checkpointer, store=store)
```

Register it in your `graphs/__init__.py`:

```python
from my_agent import build_graph

graph = build_graph(checkpointer, store)
register("my-custom-agent", graph)
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) — understand how the pieces fit together
- [Memory System](../memory/overview.md) — add persistent memory to your agent
- [Tools & Capabilities](../tools/overview.md) — register tools with risk levels and filtering
- [R0 Agent](../agents/r0-agent.md) — explore the full-featured agent implementation
