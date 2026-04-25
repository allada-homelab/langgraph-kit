# Orchestration — async tasks

Fire-and-forget background tasks tracked in the Store via
`AsyncTaskManager`. Tasks survive context compaction because they're
keyed by `(async_tasks, parent_thread_id, task_id)`. Same manager
backs the agent-callable `start_async_task` / `check_async_task` tools.

```bash
uv run python -m examples.orchestration_async_tasks
```

```python
--8<-- "examples/orchestration_async_tasks.py"
```
