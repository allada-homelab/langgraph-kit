# Memory Models

**Source:** `src/langgraph_kit/core/memory/models.py`

## MemoryType

```python
class MemoryType(str, Enum):
    USER = "user"           # Facts about the user
    FEEDBACK = "feedback"   # Behavioral guidance for the agent
    PROJECT = "project"     # Ongoing project context
    REFERENCE = "reference" # Pointers to external resources
```

## MemoryScope

```python
class MemoryScope(str, Enum):
    USER = "user"           # Belongs to a specific user
    ASSISTANT = "assistant" # Knowledge the assistant has learned
    PROJECT = "project"     # Shared across the project
    TEAM = "team"           # Shared across the team
```

## MemoryRecord

```python
class MemoryRecord(BaseModel):
    id: str                          # Unique identifier (UUID)
    title: str                       # Short descriptive title
    type: MemoryType                 # Classification
    scope: MemoryScope               # Ownership scope
    summary: str = ""                # One-line summary
    body: str = ""                   # Full content
    created_at: str = ""             # ISO timestamp
    updated_at: str = ""             # ISO timestamp
    source: str = ""                 # Origin (e.g., "auto-extraction", "user")
```

### Serialization Methods

**`to_store_value() -> dict[str, Any]`** — Serialize the record to a dict suitable for LangGraph Store's `aput()`.

**`from_store_value(cls, value: dict) -> MemoryRecord`** — Class method to deserialize from a Store value dict.

## Store Layout

Records are stored under:
```
namespace = ("memory", scope_value, type_value)
key = record.id
value = record.to_store_value()
```

For example, a user feedback record:
```
namespace = ("memory", "user", "feedback")
key = "abc-123"
value = {"id": "abc-123", "title": "No DB mocks", "type": "feedback", ...}
```
