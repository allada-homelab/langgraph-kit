# Persistent Memory Manager

**Source:** `src/langgraph_kit/core/memory/persistent.py`

A CRUD facade over LangGraph's `BaseStore` for typed memory records.

## Class: PersistentMemoryManager

### Constructor

```python
PersistentMemoryManager(store: BaseStore)
```

### Methods

#### create(record: MemoryRecord) -> None

Persist a new memory record. Stores under `("memory", record.scope, record.type)` with the record's ID as the key.

#### get(id: str, scope: MemoryScope, type: MemoryType | None = None) -> MemoryRecord | None

Retrieve a record by ID. If `type` is not specified, searches across all types within the given scope.

#### update(id: str, scope: MemoryScope, *, body: str | None = None, summary: str | None = None, type: MemoryType | None = None) -> MemoryRecord | None

Apply partial updates to an existing record. Only provided fields are changed. Updates the `updated_at` timestamp.

#### delete(id: str, scope: MemoryScope, type: MemoryType | None = None) -> bool

Remove a record by ID. Returns `True` if the record was found and deleted.

#### list_by_scope(scope: MemoryScope, type: MemoryType | None = None) -> list[MemoryRecord]

List all records in a scope, optionally filtered by type.

#### search(query: str, scope: MemoryScope, limit: int = 10) -> list[MemoryRecord]

Semantic search for records matching the query within a scope. Uses LangGraph Store's built-in search capability.

#### list_all_scopes() -> list[str]

Return the names of all scopes that contain at least one record.

## Example

```python
from langgraph_kit.core.memory.models import MemoryRecord, MemoryType, MemoryScope
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

mgr = PersistentMemoryManager(store)

# Create
record = MemoryRecord(
    id="mem-001",
    title="User prefers Go",
    type=MemoryType.USER,
    scope=MemoryScope.USER,
    summary="Deep Go expertise, new to React",
    body="User has 10 years Go experience...",
    source="auto-extraction",
)
await mgr.create(record)

# Search
results = await mgr.search("programming languages", MemoryScope.USER)

# Update
await mgr.update("mem-001", MemoryScope.USER, body="Updated content...")

# Delete
await mgr.delete("mem-001", MemoryScope.USER)
```
