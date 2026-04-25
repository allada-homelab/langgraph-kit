# Examples

Runnable demos for every user-facing subsystem of the kit. Each example
is a single Python file that you can launch with `uv run python` —
hermetically by default (no API keys), or against a real model with one
environment variable.

```bash
# Hermetic (default)
uv run python -m examples.quickstart_echo

# Real LLM
LANGGRAPH_KIT_EXAMPLES_LLM=real AGENT_LLM_API_KEY=sk-... \
    uv run python -m examples.quickstart_echo
```

## Phase 1 (this release)

- [Memory — save & recall](memory_save_recall.md) — `PersistentMemoryManager` CRUD + keyword search
- `examples/quickstart_echo.py` — minimal echo agent
- `examples/reference_deep_agent.py` — full kit stack
- `examples/tools_register_capability.py` — tool registry + risk levels
- `examples/streaming_sse_events.py` — `stream_agent_events` SSE consumer

The remaining demos are tracked in
[issue #61](https://github.com/allada-homelab/langgraph-kit/issues/61).

## How they're rendered

These docs pages pull each example's source file in via
`pymdownx.snippets`, so the rendered docs and the runnable script share
one source — change the `.py` and the docs update on the next build.

## Hermetic substrate

Hermetic mode wires the `langgraph_kit.replay.RecordedChatModel` into
every `build_llm()` call site. The same `RecordedChatModel` powers the
e2e test suite, so a demo and a test exercise identical machinery.
