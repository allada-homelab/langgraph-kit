# Public API

Top-level exports from `langgraph_kit`. These are the primary entry points for using the package.

## Configuration

```python
from langgraph_kit import AgentConfig, configure, get_config
```

| Export | Type | Description |
|--------|------|-------------|
| `AgentConfig` | `dataclass` | Frozen configuration dataclass |
| `configure(config)` | `function` | Set package-level config (call once at startup) |
| `get_config()` | `function` | Read current config |

## LLM & Persistence

```python
from langgraph_kit import build_llm, create_persistence
```

| Export | Type | Description |
|--------|------|-------------|
| `build_llm()` | `function` | Create chat model from config (auto-detects provider) |
| `create_persistence()` | `async context manager` | Yields `(checkpointer, store)` tuple |

## Agent Registry

```python
from langgraph_kit import register, get, get_dispatcher, list_agents
```

| Export | Type | Description |
|--------|------|-------------|
| `register(id, graph, *, command_dispatcher)` | `function` | Register a compiled graph |
| `get(id)` | `function` | Look up graph by ID (raises `KeyError`) |
| `get_dispatcher(id)` | `function` | Look up command dispatcher (returns `None` if absent) |
| `list_agents()` | `function` | List all registered agents as `[{"id": ..., "name": ...}]` |

## Streaming

```python
from langgraph_kit import stream_agent_events
```

| Export | Type | Description |
|--------|------|-------------|
| `stream_agent_events(graph, input, config, *, store)` | `async generator` | Stream SSE events |

## Models

```python
from langgraph_kit import ChatMessage, InvokeRequest, InvokeResponse
```

| Export | Type | Description |
|--------|------|-------------|
| `ChatMessage` | `BaseModel` | `{role, content}` |
| `InvokeRequest` | `BaseModel` | `{messages, thread_id, checkpoint_id}` |
| `InvokeResponse` | `BaseModel` | `{content, thread_id}` |

## Observability

```python
from langgraph_kit import UserInfo, build_agent_run_config
```

| Export | Type | Description |
|--------|------|-------------|
| `UserInfo` | `Protocol` | Protocol with `id` and `email` attributes |
| `build_agent_run_config(thread_id, *, user, tags, metadata)` | `function` | Build LangGraph run config |

## Subpackage Imports

For advanced usage, import directly from subpackages:

```python
# Core building blocks
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.memory.consolidation import MemoryConsolidator
from langgraph_kit.core.memory.shared import SharedMemoryManager
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.worktree import build_worktree_tools
from langgraph_kit.core.commands.dispatch import CommandDispatcher
from langgraph_kit.core.prompt_assembly.composer import PromptComposer
from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.orchestration import GENERAL_WORKERS, CODING_WORKERS

# Graph builder factories (preferred over graphs._builder)
from langgraph_kit.core.graph_builder import (
    build_command_dispatcher,
    build_middleware_stack,
    register_standard_tools,
    register_tool,
)

# Agent graphs
from langgraph_kit.graphs import register_all
from langgraph_kit.graphs.echo_agent import build_graph
from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent

# FastAPI integration
from langgraph_kit.contrib.fastapi import create_agent_router
```
