# Agent Registry

**Source:** `src/langgraph_kit/registry.py`

The registry is an in-memory mapping of agent IDs to compiled LangGraph graphs and their optional command dispatchers.

## API

### register(agent_id, graph, *, command_dispatcher=None)

```python
def register(
    agent_id: str,
    graph: Any,
    *,
    command_dispatcher: CommandDispatcher | None = None,
) -> None
```

Register a compiled graph under the given agent ID. Optionally attach a `CommandDispatcher` for slash-command support.

### get(agent_id)

```python
def get(agent_id: str) -> Any
```

Return the compiled graph for the given agent ID. Raises `KeyError` if the agent is not registered.

### get_dispatcher(agent_id)

```python
def get_dispatcher(agent_id: str) -> CommandDispatcher | None
```

Return the command dispatcher for the given agent ID, or `None` if no dispatcher was registered.

### list_agents()

```python
def list_agents() -> list[dict[str, str]]
```

Return metadata for all registered agents. Each entry has:
- `id`: The agent ID as registered
- `name`: A human-readable name derived from the ID (hyphens replaced with spaces, title-cased)

## Example

```python
from langgraph_kit import register, get, list_agents

# Register
register("my-agent", compiled_graph, command_dispatcher=dispatcher)

# Lookup
graph = get("my-agent")

# List all
agents = list_agents()
# [{"id": "my-agent", "name": "My Agent"}, ...]
```

## Registration Lifecycle

Agents are typically registered during application startup:

1. `create_persistence()` yields `(checkpointer, store)`
2. `register_all(checkpointer, store, mcp_tools)` builds and registers all built-in agents
3. Custom agents can be registered with additional `register()` calls

The registry is module-level (global state), so registered agents are available to all request handlers for the lifetime of the process.
