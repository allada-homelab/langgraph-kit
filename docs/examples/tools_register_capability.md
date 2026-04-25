# Tools — register capability

Registers three tools across `READ_ONLY`, `MUTATING`, and `DESTRUCTIVE`
risk levels, then filters the registry by `max_risk` so a worker only
sees the capabilities it's allowed to call.

```bash
uv run python -m examples.tools_register_capability
```

```python
--8<-- "examples/tools_register_capability.py"
```
