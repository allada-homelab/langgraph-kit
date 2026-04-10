# Tool Registry

**Source:** `src/langgraph_kit/core/tools/registry.py`

Manages tool capabilities with registration, multi-dimensional filtering, compilation, and prompt fragment collection.

## Class: ToolRegistry

### Methods

| Method | Description |
|--------|-------------|
| `register(cap)` | Add a single `ToolCapability` |
| `register_many(caps)` | Add multiple capabilities |
| `get(id)` | Retrieve capability by ID |
| `remove(id)` | Remove capability by ID |
| `list_all()` | Return all registered capabilities |
| `filter(*, profile, worker_type, tags, max_risk)` | Filter capabilities by criteria |
| `compile_tools()` | Return list of callable functions for LLM binding |
| `compile_tools_filtered(...)` | Filter then compile in one step |
| `collect_prompt_fragments()` | Gather `prompt_guidance` from all tools |
| `get_by_risk(risk)` | Return tools of a specific risk level |

### Filtering

```python
# Get only read-only tools for a researcher
tools = registry.filter(
    worker_type="researcher",
    max_risk=ToolRisk.READ_ONLY,
)

# Get coding-profile tools
tools = registry.filter(profile="coding")

# Compile filtered tools for LLM
callable_tools = registry.compile_tools_filtered(
    profile="coding",
    worker_type="implementer",
)
```

**Filter logic:**
- `profile` — includes tools whose `profiles` list contains the value (or is empty)
- `worker_type` — includes tools whose `worker_types` list contains the value (or is empty)
- `tags` — includes tools that have at least one matching tag
- `max_risk` — includes tools at or below the specified risk level

### Prompt Fragment Collection

```python
fragments = registry.collect_prompt_fragments()
# ["Use web search for current events...", "Memory tools store durable facts..."]
```

These fragments are injected into the system prompt by the `ToolContextProvider` during prompt assembly.

## Example

```python
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

registry = ToolRegistry()

registry.register(ToolCapability(
    id="search",
    name="Search",
    description="Search the web",
    fn=search_fn,
    risk=ToolRisk.READ_ONLY,
    profiles=["research"],
))

registry.register(ToolCapability(
    id="write-file",
    name="Write File",
    description="Write content to a file",
    fn=write_file_fn,
    risk=ToolRisk.MUTATING,
    profiles=["coding"],
))

# All tools
all_tools = registry.compile_tools()

# Only read-only tools (coordinator mode)
safe_tools = registry.compile_tools_filtered(max_risk=ToolRisk.READ_ONLY)
```
