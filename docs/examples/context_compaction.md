# Context — compaction

Builds a `PressureMonitor`, runs three synthetic conversation slices
through it (tiny / bloated with large tool outputs / past the critical
threshold), and shows how the monitor maps `pressure_pct` plus
`large_tool_outputs` to a `MitigationStrategy`. The bundled
`PressureMiddleware` consumes these signals to drive automatic
mitigation during a real run.

```bash
uv run python -m examples.context_compaction
```

```python
--8<-- "examples/context_compaction.py"
```
