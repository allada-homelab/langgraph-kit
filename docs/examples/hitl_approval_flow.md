# HITL — approval flow

Wires a tiny custom graph that calls `interrupt()` to pause for
approval, then resumes with `Command(resume={"type": "accept"})`. The
same primitive backs `ToolCapability(interrupt_before=True)` when
`AutoInterruptMiddleware` is wired in.

```bash
uv run python -m examples.hitl_approval_flow
```

```python
--8<-- "examples/hitl_approval_flow.py"
```
