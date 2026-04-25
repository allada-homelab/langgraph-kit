# Tools — risk levels + HITL gating

Pairs `ToolCapability(interrupt_before=True)` with
`AutoInterruptMiddleware`. Companion to
[HITL — approval flow](hitl_approval_flow.md): this demo shows the
static declaration side; that one shows the runtime interrupt cycle.

```bash
uv run python -m examples.tools_risk_levels_and_hitl
```

```python
--8<-- "examples/tools_risk_levels_and_hitl.py"
```
