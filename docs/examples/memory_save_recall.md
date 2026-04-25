# Memory — save & recall

Pure CRUD on the `PersistentMemoryManager`: create three typed memory
records, list them by scope + type, update one in place, and run a
keyword search. No LLM is touched — the same store-backed primitives
are what the auto-extraction middleware populates after each agent
turn.

Run it:

```bash
uv run python -m examples.memory_save_recall
```

Source (lifted verbatim from
[`examples/memory_save_recall.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/memory_save_recall.py)):

```python
--8<-- "examples/memory_save_recall.py"
```

## Why this is the canonical memory demo

- Uses `PersistentMemoryManager.create()` / `list_by_scope()` /
  `update()` / `search()` directly, so the demo doubles as living
  documentation for the public API.
- Keyword search (no embedding function configured) is the default
  fallback. To exercise semantic search, set
  `AgentConfig.memory_embedding_fn` and re-run — that flow is its own
  example in Phase 3.
- Cleanup is automatic: the `tmp_workspace()` helper drops a tempdir
  on exit, so nothing persists in the user's home or repo.
