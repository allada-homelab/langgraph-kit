# Architecture Overview

langgraph-kit is organized as a layered toolkit where each layer can be used independently or composed together into full-featured agents.

## Package Layout

```
src/langgraph_kit/
├── __init__.py              # Public API exports
├── _config.py               # AgentConfig dataclass
├── models.py                # Pydantic request/response models
├── llm.py                   # Multi-provider LLM factory
├── persistence.py           # Checkpointer + Store factory
├── observability.py         # Langfuse integration
├── streaming.py             # SSE event streaming
├── registry.py              # Agent ID → graph mapping
├── cli.py                   # Agent scaffolding CLI
├── pruning.py               # Store data cleanup
│
├── core/                    # Building blocks (composable)
│   ├── commands/            # Slash-command dispatch
│   ├── context_management/  # Pressure monitoring, compaction
│   ├── graph_builder/       # Shared agent builder factories
│   │   ├── backend.py       # CompositeBackend factory
│   │   ├── commands.py      # Command dispatcher assembly
│   │   ├── middleware.py    # Middleware stack assembly
│   │   └── tools.py         # Tool registration helpers
│   ├── memory/              # Persistent memory, consolidation, shared
│   │   ├── consolidation.py # Background memory merge/prune/normalize
│   │   └── shared.py        # Team memory sync with secret detection
│   ├── tools/               # Tool capability model + registry
│   │   └── worktree.py      # Git worktree isolation tools
│   ├── skills/              # SKILL.md discovery
│   ├── orchestration/       # Async tasks, queues, workers
│   │   ├── workers.py       # Declarative worker definitions
│   │   └── verification.py  # Coding verifier re-export
│   ├── prompt_assembly/     # Layered prompt composition
│   ├── resilience/          # Error recovery middleware
│   ├── hitl/                # Human-in-the-loop approval
│   ├── plugins/             # MCP + plugin system
│   ├── artifacts.py         # Structured UI artifacts
│   ├── coordinator.py       # Supervisor profile (read-only)
│   └── ui_events.py         # Progress, suggestions, citations
│
├── graphs/                  # Agent implementations
│   ├── __init__.py          # register_all() entry point
│   ├── _builder.py          # Re-exports from core.graph_builder
│   ├── echo_agent.py        # Minimal example
│   ├── deep_agent.py        # deepagents baseline
│   ├── r0_agent.py          # Full-featured agent
│   └── coding_agent.py      # Coding-specific agent
│
├── contrib/                 # Optional integrations
│   └── fastapi.py           # FastAPI router factory
│
└── evals/                   # Evaluation framework
    ├── models.py
    ├── runner.py
    ├── report.py
    └── metrics/
```

## Data Flow

```
User Request
    │
    ▼
┌─────────────────────────────────────────────┐
│  FastAPI Router (contrib/fastapi.py)         │
│  - Auth (CurrentUser dependency)             │
│  - Command dispatch (slash-commands)         │
│  - Message conversion                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Middleware Stack (core/)                     │
│  1. CommandMiddleware     (intercept /cmd)    │
│  2. QueuedInputMiddleware (inject queued)     │
│  3. ToolErrorMiddleware   (wrap tool calls)   │
│  4. PressureMiddleware    (monitor context)   │
│  5. ResultPersistence     (offload outputs)   │
│  6. ExtractionMiddleware  (auto-memory)       │
│  7. EmptyTurnMiddleware   (nudge model)       │
│  8. CompletionGuard       (detect premature)  │
│  9. PostRunBackstop       (final checks)      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LangGraph StateGraph                        │
│  - LLM node (via build_llm)                 │
│  - Tool execution                            │
│  - Interrupt points (HITL)                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  SSE Streaming (streaming.py)                │
│  - Token events                              │
│  - Tool call start/end                       │
│  - Artifact events (sentinel-prefixed)       │
│  - UI events (progress, suggestions)         │
│  - Interrupt events                          │
│  - [DONE] sentinel                           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
              Client / UI
```

## Persistence Model

All persistent state flows through two LangGraph primitives:

### Checkpointer

Stores conversation history, graph state, and branching metadata. Supports checkpoint-level forking for conversation branching.

- **PostgreSQL**: `AsyncPostgresSaver` — full durability
- **SQLite**: `AsyncSqliteSaver` — file-based, suitable for local dev

### Store

A hierarchical key-value store used for all application data beyond conversation messages:

| Namespace Pattern | Contents |
|-------------------|----------|
| `("memory", scope, [type])` | Persistent memory records |
| `("memory", "agent", name, [type])` | Worker-scoped agent memory |
| `("session", thread_id)` | Session notebook |
| `("async_tasks", parent_thread_id)` | Background task metadata |
| `("queue", thread_id)` | Message queue items |
| `("thread_busy",)` | Active thread locks |
| `("tool_results",)` | Cached large tool outputs |

- **PostgreSQL**: `AsyncPostgresStore` — full persistence with semantic search
- **SQLite fallback**: `InMemoryStore` — data lost on restart

## Composition Model

langgraph-kit follows a **composition over inheritance** approach. Each subsystem is independent:

```python
# Use just memory
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

# Use just tools
from langgraph_kit.core.tools.registry import ToolRegistry

# Use just prompt assembly
from langgraph_kit.core.prompt_assembly.composer import PromptComposer

# Or compose everything via the graph builder
from langgraph_kit.core.graph_builder import (
    register_standard_tools,
    build_middleware_stack,
    build_command_dispatcher,
)
```

The `core.graph_builder` package provides convenience functions that wire subsystems together, but you can always compose them manually for full control.

## Agent Contract

Every agent graph must implement this function signature:

```python
def build_graph(checkpointer: Any, store: Any) -> CompiledStateGraph:
    """Build and return a compiled LangGraph graph."""
    ...
```

The graph is registered with an ID and optionally a command dispatcher:

```python
from langgraph_kit import register
register("my-agent", graph, command_dispatcher=dispatcher)
```

## Extension Points

| Extension | Mechanism | Example |
|-----------|-----------|---------|
| New agent | Implement `build_graph()` | `graphs/my_agent.py` |
| New tool | Register `ToolCapability` | `registry.register(cap)` |
| New command | Register handler on dispatcher | `dispatcher.register("/foo", handler)` |
| New prompt section | Add to `SectionRegistry` | `sections.register(PromptSection(...))` |
| New context provider | Implement `ContextProvider` protocol | `class MyProvider: async def provide(...)` |
| New middleware | Subclass `_AgentMiddleware` | `class MyMiddleware(_AgentMiddleware)` |
| New skill | Write SKILL.md file | `skills/my_skill/SKILL.md` |
| MCP tools | Configure `mcp_servers` JSON | `AgentConfig(mcp_servers='...')` |
| Python plugins | Write `contribute()` function | `plugins/my_plugin.py` |
