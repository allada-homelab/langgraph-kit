# Agent Memory

**Source:** `src/langgraph_kit/core/memory/agent_memory.py`

Worker-scoped memory for specialized agents (e.g., researcher, implementer, verifier) to retain role-specific knowledge.

## Class: AgentMemoryManager

### Constructor

```python
AgentMemoryManager(store: BaseStore, agent_name: str)
```

The `agent_name` determines the namespace prefix: `("memory", "agent", agent_name, [type])`.

### Methods

Same CRUD API as `PersistentMemoryManager`:

- `create(record)` — Persist a new record
- `get(id, type=None)` — Retrieve by ID
- `update(id, *, body=None, summary=None, type=None)` — Partial update
- `delete(id, type=None)` — Remove a record
- `list_by_type(type=None)` — List records, optionally filtered
- `search(query, limit=10)` — Semantic search
- `snapshot_from(records)` — Initialize agent memory from source records

### snapshot_from(records)

Bulk-initialize the agent's memory from a list of existing `MemoryRecord` objects. Useful for bootstrapping a worker agent with relevant context from the parent conversation.

## Namespace Layout

```
("memory", "agent", "researcher", "reference")  → researcher's reference memories
("memory", "agent", "implementer", "project")   → implementer's project notes
("memory", "agent", "verifier", "feedback")      → verifier's learned patterns
```

## Use Case

In multi-agent orchestration, each worker agent operates in its own context. Agent memory allows workers to build up specialized knowledge:

- A **researcher** might remember preferred search patterns and documentation URLs
- An **implementer** might remember architecture conventions and file locations
- A **verifier** might remember testing patterns that caught bugs previously

This knowledge persists across conversations and is scoped to the worker role, not the thread.
