# Memory Consolidation

**Source:** `src/langgraph_kit/core/memory/consolidation.py`

Background memory maintenance that uses an LLM to merge duplicates, prune stale entries, and normalize records.

## Class: MemoryConsolidator

### Constructor

```python
MemoryConsolidator(memory_manager: PersistentMemoryManager, llm: Any)
```

### Methods

#### consolidate(scope=MemoryScope.USER, limit=100) -> ConsolidationResult

Run a consolidation pass on memories in the given scope:

1. Loads all memories in the scope (up to `limit`)
2. Formats them into a prompt for the LLM
3. LLM identifies actions: keep, delete, merge, update
4. Applies the actions to the memory store
5. Returns a result summary

Skips consolidation if fewer than 2 records exist in the scope.

## ConsolidationResult

```python
class ConsolidationResult:
    kept: int           # Records left unchanged
    deleted: int        # Records removed
    merged: int         # Record groups merged into one
    updated: int        # Records modified
    errors: list[str]   # Errors encountered during apply

    @property
    def total_actions(self) -> int: ...
```

## LLM Actions

The consolidation LLM produces a JSON array of actions:

| Action | Fields | Description |
|--------|--------|-------------|
| `keep` | `id` | Leave record unchanged |
| `delete` | `id`, `reason` | Remove a stale or irrelevant record |
| `merge` | `source_ids`, `merged` | Combine duplicates into a single record |
| `update` | `id`, `updates` | Modify a record's body or summary |

### Merge Action Detail

```json
{
    "action": "merge",
    "source_ids": ["mem-001", "mem-002"],
    "merged": {
        "title": "Combined title",
        "type": "user",
        "summary": "...",
        "body": "..."
    }
}
```

Source records are deleted and a new merged record is created with `source="consolidation_merge"`.

## Consolidation Rules

The LLM is instructed to be **conservative**:

- Merge near-duplicates into a single, better record
- Delete records that are stale, no longer relevant, or derivable from the environment
- Update records that need correction or clarification
- Keep records that are accurate and useful as-is
- Never invent new facts — only reorganize existing ones

## Usage

```python
from langgraph_kit.core.memory.consolidation import MemoryConsolidator

consolidator = MemoryConsolidator(memory_mgr, llm)
result = await consolidator.consolidate(scope=MemoryScope.USER)
print(result)
# ConsolidationResult(kept=8, deleted=2, merged=1, updated=1, errors=[])
```

Consolidation is typically run as a background maintenance task, not during normal agent operation.
