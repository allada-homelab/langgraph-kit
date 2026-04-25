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

#### search(query: str, scope: MemoryScope, memory_type: MemoryType | None = None, limit: int = 5) -> list[MemoryRecord]

Returns records most relevant to `query` within a scope. The ranking mode
is chosen by whether an embedding function was supplied — either directly
via the `embedding_fn=` constructor arg or through
`AgentConfig.memory_embedding_fn`:

- **Embedding-backed (semantic)**: the query is embedded, every candidate
  vector in the scope's embedding namespace is compared by cosine
  similarity, and the top-K matching records are fetched. Records are
  indexed automatically on `create` and on any `update` that changes
  `title`, `summary`, `body`, or `scope`.
- **Keyword (default)**: records in the scope are scored by
  case-insensitive token overlap of `title + summary + body` against the
  query tokens. No vectors, no embedding calls.

There is **no silent fallback** between the two. The presence of the
embedding callable is the only switch, so behaviour is deterministic
regardless of which `Store` backend is used.

```python
# Keyword mode (default): no embedding function configured.
mgr = PersistentMemoryManager(store)
await mgr.search("python", MemoryScope.USER)  # ranks by token overlap

# Semantic mode: opt in by passing a batch embedding function.
async def embed(texts: list[str]) -> list[list[float]]:
    # OpenAI, fastembed, local model — any async batch embedder.
    ...

mgr = PersistentMemoryManager(store, embedding_fn=embed)
await mgr.search("python", MemoryScope.USER)  # ranks by cosine similarity
```

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
