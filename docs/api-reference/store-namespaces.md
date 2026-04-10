# Store Namespaces

Complete inventory of LangGraph Store namespaces used by langgraph-kit.

All persistent data (beyond conversation messages in the checkpointer) is stored in the LangGraph Store under hierarchical tuple-based namespaces.

## Namespace Map

### Memory

| Namespace | Key | Value | Module |
|-----------|-----|-------|--------|
| `("memory", scope, type)` | record ID | `MemoryRecord.to_store_value()` | `core/memory/persistent.py` |
| `("memory", "agent", agent_name, type)` | record ID | `MemoryRecord.to_store_value()` | `core/memory/agent_memory.py` |

**Examples:**
```
("memory", "user", "feedback")  → user feedback records
("memory", "project", "reference")  → project reference records
("memory", "agent", "researcher", "user")  → researcher's user knowledge
```

### Session

| Namespace | Key | Value | Module |
|-----------|-----|-------|--------|
| `("session", thread_id)` | `"notebook"` | Notebook content string | `core/memory/session.py` |

### Orchestration

| Namespace | Key | Value | Module |
|-----------|-----|-------|--------|
| `("async_tasks", parent_thread_id, task_id)` | task_id | `AsyncTask.model_dump()` | `core/orchestration/async_tasks.py` |
| `("queue", thread_id)` | item ID | `QueuedItem.model_dump()` | `core/orchestration/queue.py` |
| `("thread_busy",)` | thread_id | `{"busy": True, "timestamp": ...}` | `core/orchestration/queue.py` |

### Tool Results

| Namespace | Key | Value | Module |
|-----------|-----|-------|--------|
| `("tool_results",)` | result ID | `{"content": ..., "timestamp": ...}` | `core/context_management/result_persistence.py` |

### Agent Backends (deepagents)

| Namespace | Key | Value | Module |
|-----------|-----|-------|--------|
| `(agent_name, "memories")` | item ID | Varies | `core/graph_builder/backend.py` |
| `(agent_name, "notes")` | item ID | Varies | `core/graph_builder/backend.py` |

## Pruning

The `pruning.py` module cleans up stale data from these namespaces:

| Namespace | Pruning Rule |
|-----------|-------------|
| `("tool_results",)` | Items older than `max_age_seconds` (default 7 days) |
| `("thread_busy",)` | Locks older than 600 seconds (auto-expire) |

```python
from langgraph_kit.pruning import prune_store

result = await prune_store(store, max_age_seconds=86400 * 7)
print(f"Deleted {result.tool_results_deleted} old tool results")
print(f"Cleared {result.stale_locks_cleared} stale locks")
```
