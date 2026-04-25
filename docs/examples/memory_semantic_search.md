# Memory — semantic search

Wires `PersistentMemoryManager` with an `embedding_fn` so `search()`
ranks by cosine similarity. The demo uses a toy bag-of-words embedder
to stay hermetic; swap in any async embedding API for production.

```bash
uv run python -m examples.memory_semantic_search
```

```python
--8<-- "examples/memory_semantic_search.py"
```
