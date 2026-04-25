# Memory — consolidation

Merges near-duplicate memory records via `MemoryConsolidator`. Wires a
scripted JSON response so the demo runs hermetically; production
consolidation calls the LLM to decide between
`keep` / `delete` / `merge` / `update`.

```bash
uv run python -m examples.memory_consolidation
```

```python
--8<-- "examples/memory_consolidation.py"
```
