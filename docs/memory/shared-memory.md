# Shared Memory

**Source:** `src/langgraph_kit/core/memory/shared.py`

Team/shared memory management with secret detection and cross-scope synchronization.

## Class: SharedMemoryManager

### Constructor

```python
SharedMemoryManager(memory_manager: PersistentMemoryManager)
```

### Methods

#### publish_to_team(record, *, allow_all_types=False) -> MemoryRecord

Publish a memory record to the team scope after validation.

**Validation steps:**
1. **Type check** — only `PROJECT` and `REFERENCE` types are shareable by default. Pass `allow_all_types=True` to override.
2. **Secret scan** — scans title, summary, and body for secret patterns. Raises `SecretDetectedError` if suspected secrets are found.
3. **Creation** — creates a copy of the record in `MemoryScope.TEAM` with `source="published_from:{scope}:{id}"`.

**Raises:**
- `SecretDetectedError` — if the record contains suspected secret material
- `ValueError` — if the memory type is not shareable

#### sync_from_team(target_scope=MemoryScope.PROJECT, limit=50) -> list[MemoryRecord]

Pull team memories into a target scope. Only copies records that don't already exist in the target (matched by title + type for deduplication).

#### list_team_memories(memory_type=None, limit=50) -> list[MemoryRecord]

List memories in the team scope, optionally filtered by type.

#### scan_for_secrets(text) -> list[str]

Check text for patterns that look like secrets. Returns a list of matched pattern descriptions. Empty if clean.

## Secret Detection

The manager scans for these patterns before allowing team publishing:

| Pattern | Example |
|---------|---------|
| API key assignments | `api_key=sk-abc123` |
| Secret/token/password assignments | `secret=mysecret` |
| Bearer tokens | `Bearer eyJhbG...` |
| PEM private keys | `-----BEGIN PRIVATE KEY-----` |
| GitHub PATs | `ghp_ABC123...` |
| OpenAI-style keys | `sk-ABC123...` |
| AWS access keys | `AKIA...` |

## Shareable Types

By default, only these types can be published to team scope:

- `MemoryType.PROJECT` — project context and decisions
- `MemoryType.REFERENCE` — pointers to external resources

`USER` and `FEEDBACK` types are blocked by default since they may contain personal preferences. Use `allow_all_types=True` to override.

## SecretDetectedError

```python
class SecretDetectedError(Exception):
    """Raised when a memory contains suspected secret material."""
```

## Usage

```python
from langgraph_kit.core.memory.shared import SharedMemoryManager, SecretDetectedError

shared = SharedMemoryManager(memory_mgr)

# Publish a project memory to team scope
try:
    team_record = await shared.publish_to_team(project_record)
except SecretDetectedError:
    print("Record contains secrets, cannot publish")

# Sync team memories into project scope
synced = await shared.sync_from_team(target_scope=MemoryScope.PROJECT)
print(f"Synced {len(synced)} records from team")
```
