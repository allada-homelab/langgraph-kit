# Persistence

**Source:** `src/langgraph_kit/persistence.py`

The persistence module provides an async context manager that yields a `(checkpointer, store)` tuple, with automatic backend selection based on the database URL.

## API

### create_persistence()

```python
@asynccontextmanager
async def create_persistence() -> AsyncGenerator[tuple[Any, Any]]:
```

Yields a `(checkpointer, store)` tuple. Both are initialized and ready to use. Resources are cleaned up when the context manager exits.

## Backend Selection

| URL Prefix | Checkpointer | Store | Data Durability |
|------------|-------------|-------|-----------------|
| `postgresql://` | `AsyncPostgresSaver` | `AsyncPostgresStore` | Full persistence |
| `sqlite:///path` | `AsyncSqliteSaver` | `InMemoryStore` | Checkpoints persisted; store data lost on restart |

### PostgreSQL

```python
async with create_persistence() as (checkpointer, store):
    # Both backed by PostgreSQL
    # Tables auto-created via setup()
    ...
```

LangGraph checkpoint tables are auto-created in the database. No Alembic migration is needed — the `setup()` call handles schema initialization.

The `postgresql+psycopg` scheme (common in SQLAlchemy URLs) is automatically normalized to plain `postgresql` for LangGraph compatibility.

### SQLite

```python
async with create_persistence() as (checkpointer, store):
    # Checkpoints in SQLite file
    # Store is InMemoryStore (ephemeral)
    ...
```

The SQLite path is extracted from the URL (e.g., `sqlite:///checkpoints.db` → `checkpoints.db`). Falls back to `checkpoints.db` if the path is empty.

**Important:** With SQLite, the Store is `InMemoryStore` — all store data (memories, queue items, session notebooks) is lost when the process restarts. Use PostgreSQL for production.

## Usage with Agent Registration

```python
from langgraph_kit import create_persistence
from langgraph_kit.graphs import register_all

async with create_persistence() as (checkpointer, store):
    await register_all(checkpointer, store, mcp_tools=[])
    # Agents are now registered and ready to serve requests
    ...
```

## What Gets Stored Where

### Checkpointer (conversation state)

- Message history per thread
- Graph state snapshots (checkpoints)
- Branching metadata for conversation forking
- Interrupt state for HITL workflows

### Store (application data)

- Persistent memory records
- Session notebooks
- Async task metadata
- Message queues
- Thread busy locks
- Cached tool outputs

See [Store Namespaces](../api-reference/store-namespaces.md) for the complete namespace inventory.
