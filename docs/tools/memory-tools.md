# Memory Tools

**Source:** `src/langgraph_kit/core/tools/memory_tools.py`

Agent-callable tools for managing persistent memory through the `PersistentMemoryManager`.

## build_memory_tools(memory_manager)

Returns a list of 5 async tool functions:

### save_memory(title, type, scope, summary, body)

Create a new persistent memory record.

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Short descriptive title |
| `type` | `str` | Memory type: user, feedback, project, reference |
| `scope` | `str` | Memory scope: user, assistant, project, team |
| `summary` | `str` | One-line summary |
| `body` | `str` | Full content |

### list_memories(scope, type=None)

List stored memory records by scope, optionally filtered by type.

### search_memories(query, scope)

Semantic search for relevant memories within a scope.

### update_memory(id, scope, body=None, summary=None)

Update an existing memory record. Only provided fields are changed.

### delete_memory(id, scope)

Delete a memory record by ID and scope.

## Registration

Memory tools are typically registered via the builder utility:

```python
from langgraph_kit.core.graph_builder.tools import register_memory_tools

register_memory_tools(tool_registry, memory_manager)
```

This wraps each function as a `ToolCapability` with appropriate metadata:
- `save_memory` and `update_memory` → `ToolRisk.MUTATING`
- `delete_memory` → `ToolRisk.DESTRUCTIVE`
- `list_memories` and `search_memories` → `ToolRisk.READ_ONLY`
