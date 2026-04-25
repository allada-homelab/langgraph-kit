# Orchestration — workers

Inspects the kit's pre-composed `GENERAL_WORKERS` and `CODING_WORKERS`
worker definitions and shows how to extend them with a domain-specific
sub-agent. The `reference_deep_agent` and `coding_agent` builders both
consume these lists at graph-build time.

```bash
uv run python -m examples.orchestration_workers
```

```python
--8<-- "examples/orchestration_workers.py"
```
