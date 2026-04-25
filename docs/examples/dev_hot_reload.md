# Dev — hot reload watcher

`Reloader` polls a directory at a fast cadence and emits `FileChange`
records on add / modify / remove. Foundation for the `langgraph-kit
dev` server (graph rebuild + checkpoint preservation + inspector UI)
tracked in #36.

```bash
uv run python -m examples.dev_hot_reload
```

```python
--8<-- "examples/dev_hot_reload.py"
```
