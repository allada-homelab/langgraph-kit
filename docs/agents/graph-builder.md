# Graph Builder

**Source:** `src/langgraph_kit/core/graph_builder/`

Shared factories for assembling agent graph infrastructure: tool registration, middleware stacks, command dispatchers, and backend configuration. Extracted from the original `graphs/_builder.py` into a proper core subpackage.

> **Note:** `graphs/_builder.py` still exists as a backwards-compatible re-export shim. New code should import from `langgraph_kit.core.graph_builder`.

## Package Structure

```
core/graph_builder/
├── __init__.py      # Public exports
├── backend.py       # CompositeBackend factory for deepagents
├── commands.py      # Command dispatcher assembly
├── middleware.py     # Middleware stack assembly
└── tools.py         # Tool registration helpers
```

## Public API

```python
from langgraph_kit.core.graph_builder import (
    build_command_dispatcher,
    build_middleware_stack,
    register_standard_tools,
    register_tool,
)
```

---

## tools.py — Tool Registration

### register_tool(registry, tool_fn, *, id_prefix, name, tags, risk, prompt_guidance)

Low-level helper that wraps a function as a `ToolCapability` and registers it.

```python
register_tool(
    registry,
    my_fn,
    id_prefix="custom",
    name="my_tool",
    tags=["search"],
    risk=ToolRisk.READ_ONLY,
    prompt_guidance="Use this tool when...",
)
```

### Individual Registration Functions

| Function | Tools Registered |
|----------|-----------------|
| `register_memory_tools(registry, mgr)` | save, list, search, update, delete memory |
| `register_retrieval_tool(registry, store)` | Retrieve persisted large tool outputs |
| `register_search_tool(registry)` | Deferred tool search (returns `DeferredToolRegistry`) |
| `register_skill_tools(registry, skills_dir?)` | discover_skills, get_skill_guidance |
| `register_async_tools(registry, store, *, parent_thread_id)` | start, check, cancel, list async tasks |
| `register_ui_tools(registry)` | create_artifact, emit_progress, suggest_actions, add_citation |
| `register_hitl_tools(registry)` | approve_action |

### register_standard_tools(registry, memory_mgr, store, *, parent_thread_id, mcp_tools)

Registers the full standard tool suite by calling all individual registration functions, plus any MCP tools passed in.

---

## middleware.py — Middleware Stack

### build_middleware_stack(*, llm, memory_mgr, pressure_monitor, command_dispatcher)

```python
def build_middleware_stack(
    *,
    llm: Any,
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    command_dispatcher: CommandDispatcher | None = None,
) -> tuple[list[Any], PressureMonitor]:
```

Returns `(middleware_list, pressure_monitor)`:

```python
[
    CommandMiddleware(dispatcher),        # if dispatcher provided
    RuntimeStateMiddleware(),
    QueuedInputMiddleware(),
    ToolErrorMiddleware(max_retries=1),
    PressureMiddleware(pressure_monitor),
    ResultPersistenceMiddleware(),
    ExtractionMiddleware(AutoMemoryExtractor(memory_mgr, llm), scope=MemoryScope.USER),
    EmptyTurnMiddleware(max_nudges=2),
    CompletionGuardMiddleware(min_tool_calls=1),
    StopHooksMiddleware(),
    PostRunBackstopMiddleware(),
]
```

---

## commands.py — Command Dispatcher

### build_command_dispatcher(memory_mgr, pressure_monitor, tool_registry?)

```python
def build_command_dispatcher(
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    tool_registry: ToolRegistry | None = None,
) -> CommandDispatcher:
```

Returns a `CommandDispatcher` with built-in commands registered:
`/help`, `/memory`, `/context`, `/compact`, `/status`, `/tools` (if registry provided), `/skills`.

---

## backend.py — Backend Factory

### build_backend_factory(agent_name)

```python
def build_backend_factory(agent_name: str) -> Callable
```

Creates a `CompositeBackend` factory for deepagents with agent-specific Store namespaces:

| Route | Backend | Namespace |
|-------|---------|-----------|
| `/memories/` | StoreBackend | `(agent_name, "memories")` |
| `/notes/` | StoreBackend | `(agent_name, "notes")` |
| _(default)_ | StateBackend | _(ephemeral per-thread scratch)_ |

---

## Usage in Agent Builders

```python
from langgraph_kit.core.graph_builder import (
    build_command_dispatcher,
    build_middleware_stack,
    register_standard_tools,
)

memory_mgr = PersistentMemoryManager(store)
tool_registry = ToolRegistry()
pressure_monitor = PressureMonitor()

register_standard_tools(tool_registry, memory_mgr, store, parent_thread_id=thread_id)
dispatcher = build_command_dispatcher(memory_mgr, pressure_monitor, tool_registry)
middleware, _ = build_middleware_stack(
    llm=llm,
    memory_mgr=memory_mgr,
    pressure_monitor=pressure_monitor,
    command_dispatcher=dispatcher,
)
```

## Backwards Compatibility

`graphs/_builder.py` re-exports everything from `core.graph_builder`:

```python
# Still works, but prefer the new import path
from langgraph_kit.graphs._builder import build_middleware_stack

# Preferred
from langgraph_kit.core.graph_builder import build_middleware_stack
```
