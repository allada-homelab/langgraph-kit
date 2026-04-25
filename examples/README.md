# langgraph-kit examples

Runnable demos for every user-facing subsystem of the kit. Each example
is a single Python file that you can launch with `uv run python` —
hermetically by default, no API keys required.

## How to run

```bash
# Hermetic (default) — uses RecordedChatModel-backed scripted LLMs.
uv run python -m examples.quickstart_echo

# Real LLM — needs AGENT_LLM_API_KEY.
LANGGRAPH_KIT_EXAMPLES_LLM=real AGENT_LLM_API_KEY=sk-... \
    uv run python -m examples.quickstart_echo

# Run the whole hermetic smoke suite (used in CI).
just examples-smoke
```

## What's here

| File | Showcases |
| --- | --- |
| [`quickstart_echo.py`](quickstart_echo.py) | The minimal `build_graph(checkpointer, store)` contract |
| [`basic_deep_agent.py`](basic_deep_agent.py) | `build_basic_deep_agent` — deepagents framework, no kit features |
| [`reference_deep_agent.py`](reference_deep_agent.py) | The full kit stack wired together |
| [`memory_save_recall.py`](memory_save_recall.py) | `PersistentMemoryManager` CRUD + keyword search |
| [`tools_register_capability.py`](tools_register_capability.py) | `ToolCapability` with risk levels + filtering |
| [`streaming_sse_events.py`](streaming_sse_events.py) | Consuming `stream_agent_events`'s SSE output |
| [`prompt_assembly_sections.py`](prompt_assembly_sections.py) | `SectionRegistry` + `PromptComposer` cache-aware ordering |
| [`context_compaction.py`](context_compaction.py) | `PressureMonitor` thresholds + mitigation strategies |
| [`orchestration_workers.py`](orchestration_workers.py) | Worker definitions + extending the catalogue |
| [`hitl_approval_flow.py`](hitl_approval_flow.py) | `interrupt()` → `Command(resume=...)` round-trip |
| [`fastapi_minimal_router.py`](fastapi_minimal_router.py) | Mounting the agent router via ASGI in-process client |
| [`replay_record_and_play.py`](replay_record_and_play.py) | Recording a hermetic run + reloading the JSON |

The remaining demos (security, audit, DR, rate limit, plugins, MCP,
observability, evals, full FastAPI) ship in Phase 3 — tracked in
[issue #61](https://github.com/allada-homelab/langgraph-kit/issues/61).

## CLI shortcut

After `pip install` (from a source checkout), examples are also
discoverable from the kit's CLI:

```bash
uv run python -m langgraph_kit.cli examples list
uv run python -m langgraph_kit.cli examples run quickstart_echo
uv run python -m langgraph_kit.cli examples run quickstart_echo --real-llm
```

## Hermetic vs real LLM

Examples default to a scripted LLM provided by
`langgraph_kit.replay.RecordedChatModel`. Set
`LANGGRAPH_KIT_EXAMPLES_LLM=real` to flip them onto the real model.

In real-LLM mode the helper in [`_lib.py`](_lib.py) installs an
`AgentConfig` pinned to `claude-haiku-4-5` (cheap) and routes the
database URL into a temporary directory that auto-cleans on exit. Caps
on output tokens / turns are enforced for any example that opts in to
real-LLM execution to keep cost predictable.

## Smoke-test runner

[`run_all.py`](run_all.py) discovers every `examples/*.py` (excluding
`_lib.py`, `run_all.py`, and anything starting with `_`) and runs each
in a fresh subprocess with a 60-second timeout. Examples that need
external services (FastAPI server, MCP server, Langfuse) declare
`REQUIRES_NETWORK = True` at module top and are skipped unless the
runner exports `RUN_NETWORK=1` (used by the nightly workflow, not by
per-PR CI).

## How to add an example

1. Pick a feature that doesn't yet have a demo — see the table above
   plus the open Phase 2 / 3 sub-issues.
2. Create `examples/<name>.py`. Use this template:

   ```python
   """<Feature name> — one-paragraph pitch.

   What this shows
   ---------------
   - Concrete capability 1
   - Concrete capability 2

   How to run
   ----------
       uv run python -m examples.<name>

   Expected output
   ---------------
   <3-5 lines of representative stdout>
   """

   from __future__ import annotations

   import asyncio

   from examples._lib import (
       banner, hermetic, line, make_in_memory_persistence,
       patch_build_llm, scripted_llm, tmp_workspace, answer,
   )


   async def main() -> None:
       banner("<name>")
       with tmp_workspace() as workspace:
           if hermetic():
               with patch_build_llm(scripted_llm([answer("...")])):
                   await _run(workspace)
           else:
               from examples._lib import configure_real_llm
               configure_real_llm(workspace)
               await _run(workspace)


   async def _run(workspace: object) -> None:
       _ = workspace
       # ... the actual demo ...


   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. Persist any state through `tmp_workspace()` — never `~` or repo root.
4. Run `just examples-smoke` to confirm it stays green; CI runs the same
   command on every PR.
