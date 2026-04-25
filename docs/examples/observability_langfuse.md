# Observability — Langfuse tracing (real-LLM)

Marks `REQUIRES_NETWORK = True`, so the per-PR hermetic tier skips it.
The nightly real-LLM workflow runs it against `claude-haiku-4-5` and
posts a trace to Langfuse if `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` are set.

```bash
LANGGRAPH_KIT_EXAMPLES_LLM=real \
    AGENT_LLM_API_KEY=sk-... \
    LANGFUSE_HOST=https://cloud.langfuse.com \
    LANGFUSE_PUBLIC_KEY=pk-... \
    LANGFUSE_SECRET_KEY=sk-... \
    uv run python -m examples.observability_langfuse
```

```python
--8<-- "examples/observability_langfuse.py"
```
