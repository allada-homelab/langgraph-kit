# Evals — rule-based + model-graded

Rule-based metrics (`ResponseLengthMetric`, `LatencyMetric`,
`ErrorFreeMetric`) score a synthetic `TraceData` hermetically.
`LLMJudgeMetric` activates with `LANGGRAPH_KIT_EXAMPLES_LLM=real`.

```bash
uv run python -m examples.evals_rule_and_model_graded
```

```python
--8<-- "examples/evals_rule_and_model_graded.py"
```
