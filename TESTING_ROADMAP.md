# Testing Roadmap

**Status as of 2026-04-24:** **Phase 4 complete. 765 tests passing. 90.55% overall line coverage** (from 72% baseline). Regression guard verified — reverting commit `1383151` (the deferred_tools gating fix) causes `test_empty_deferred_registry_does_not_push_llm_toward_tool_search` and `test_default_build_does_not_activate_deferred_tools_condition` to fail, restoring the fix returns them to green. The e2e layer demonstrably catches the bug class it was built for. New work this session added 281 tests across 23 new files: coverage fills for evals (rule_based, model_graded, report, runner), agui streaming + contrib, mcp server/client with mocked modules, worktree_tools, async_task_manager, replay_runner, observability, persistence, plugin_loader, llm_factory, cost_callback, tracing_storage, shared_memory, consolidation, agent_memory; plus e2e coverage for search_memories tool variants, deferred-tool argument-validation edges, prompt-composer STABLE/VOLATILE priority + provider-failure isolation, tool-error retry semantics, streaming-mode builder, graph-registration register_all, and skills/async_tasks/extensions ACTIVATION_SECTIONS invariants. One kit bug surfaced and fixed: `src/langgraph_kit/contrib/a2a.py` had `from fastapi import Request` scoped inside `create_a2a_router()` — combined with `from __future__ import annotations`, FastAPI's runtime type-hint resolver couldn't see `Request` and misclassified `request: Request` route parameters as query params, 422-ing every A2A endpoint at call time. Moved the import to module scope with a comment explaining why it must stay there and a `# noqa: TC002` because basedpyright would otherwise suggest moving it back to TYPE_CHECKING.

## Goal

Add an end-to-end test layer to langgraph-kit that catches composition-level bugs unit tests can't see — the class the recent `deferred_tools` spinning regression belongs to, where every unit was individually correct but the composition was broken. Grow that layer systematically, feature by feature, until coverage is near-complete.

## Context (short version)

Unit tests cover ~70% of the kit; single-component integration tests cover ~25%; essentially zero tests run a real compiled graph with an LLM in the loop. `build_*` tests all replace `deepagents.create_deep_agent` with a `MagicMock` and only inspect call args. No test in the suite would have caught the `deferred_tools` bug (empty registry + "search first" prompt → LLM spins on `tool_search`). The new layer fixes that by running real graphs against a scripted LLM with in-memory stores.

Full rationale lives in the commit that introduced this file.

## Approach (summary)

- Reuse `src/langgraph_kit/replay/player.py:RecordedChatModel` as the scripted LLM.
- Reuse `tests/conftest.py:MockStore` as the store; use `langgraph.checkpoint.memory.InMemorySaver` as the checkpointer.
- Patch `langgraph_kit.graphs._builder.build_llm` to inject the scripted model (already the established pattern in ~15 existing tests).
- Do **not** mock `deepagents.create_deep_agent` — the compiled graph is the thing under test.
- Tests live under `tests/e2e/` with a new `e2e` pytest marker. Marker is not a gate — e2e tests run by default (they're fast).
- Shared fixtures in `tests/e2e/conftest.py`, scripting helpers in `tests/e2e/helpers.py`.

---

## Phase checklists

### Phase 0 — commit this roadmap

- [x] Create `TESTING_ROADMAP.md` at repo root
- [x] Commit on branch `testing/e2e-roadmap` and open PR

### Phase 1 — spike (validate the critical assumption)

Prove a `RecordedChatModel` can drive a real kit-built graph through a multi-turn conversation with tool calls. If this fails, the rest of the plan is wrong — stop and re-scope.

- [x] Write `tests/e2e/test_spike.py` — build `build_reference_deep_agent` with `RecordedChatModel` scripted for (turn 1: tool call → turn 2: final content); `ainvoke`; assert tool call landed in state and final content reached the user
- [x] Run it (`uv run pytest tests/e2e/test_spike.py -v`) — passes
- [x] Delete `test_spike.py` (cleanup — spike only existed to de-risk the approach)
- [x] Fix surfaced gap: `RecordedChatModel.bind_tools` was missing, making it unusable against `create_agent`. Added as pass-through override with a unit test in `test_integration_features.py`.
- [x] Noted: deepagents v0.6 `DeprecationWarning`s (`backend=` factory, `StateBackend(runtime)`) fire during real graph invocation. Filter globally in `tests/e2e/conftest.py` in Phase 2.

### Phase 2 — infrastructure

- [x] Add `e2e` marker to `pyproject.toml` under `[tool.pytest.ini_options].markers`
- [x] Create `tests/e2e/__init__.py`
- [x] Create `tests/e2e/conftest.py` with fixtures:
  - [x] `checkpointer` → `InMemorySaver()` per test
  - [x] `e2e_store` → reuses `MockStore` from root `tests/conftest.py`
  - [x] `patched_build_llm` → context-manager factory that patches `langgraph_kit.graphs._builder.build_llm`
  - [x] autouse fixture filtering deepagents v0.6 `DeprecationWarning`s so the kit-side migration stays out of scope
- [x] Create `tests/e2e/helpers.py` with:
  - [x] `scripted_llm(turns)` — wraps `ConversationRecording` + `LLMInteraction` construction
  - [x] `tool_call_turn(name, args=None, call_id=None)` — builder for single-tool-call `output_message` dicts
  - [x] `multi_tool_call_turn(calls)` — builder for multi-tool-call `output_message` dicts
  - [x] `answer(content)` — builder for final-response `output_message` dicts
  - [x] `assert_tool_invoked(state, tool_name)` — inspects final state's messages
  - [x] `last_ai_message(state)` — assertion helper
- [x] Smoke test (`tests/e2e/test_infrastructure_smoke.py`) confirms fixtures + helpers compose against a real reference agent graph

### Phase 3 — MVP flagship scenarios

Four files, six tests total. Depth over breadth — each test catches a distinct class of composition bug.

- [x] `tests/e2e/test_deferred_tools_e2e.py`
  - [x] `test_deferred_tool_discoverable_and_callable` — populated registry; script `tool_search` → `call_deferred_tool` → final; assert deferred tool actually ran and its `ToolMessage` reached the LLM
  - [x] `test_empty_deferred_registry_does_not_push_llm_toward_tool_search` — default build with empty registry; script one user turn via `capturing_scripted_llm`; assert the system prompt the LLM *received* does NOT contain the `deferred_tools_awareness` marker (direct regression guard — would have caught the original bug)
  - [x] `test_tool_loop_guard_advises_when_llm_spins_on_tool_search` — populated registry; LLM scripted to call `tool_search` 6× in a row; assert `ToolLoopGuardMiddleware`'s advisory message appears
  - [x] **Bug surfaced:** `ToolLoopGuardMiddleware` kept its streak counter in a `ContextVar[dict]` set via `.set(new_dict)`, which is copy-on-write per asyncio task. LangGraph schedules each tool call as its own task, so every call saw `count=1` and the guard never fired under real execution. Unit tests missed this because they ran every call in a single coroutine. Fixed by keying on `thread_id` via a module-level dict (extracted from `runtime.execution_info.thread_id` in `abefore_agent`/`aafter_agent` and `request.runtime.config["configurable"]["thread_id"]` in `awrap_tool_call`). Added two unit tests (`test_streak_persists_across_sibling_asyncio_tasks`, `test_streak_isolated_between_concurrent_runs`) to guard against regressions of this class.
- [x] `tests/e2e/test_middleware_ordering_e2e.py`
  - [x] `test_save_memory_tool_persists_before_stop_hook_runs` — asserts tool-call persistence + stop-hook ordering: save_memory tool actually reaches MockStore, AND the stop hook's captured state contains the save_memory ToolMessage (i.e. hook ran after tool execution) — stop hook captures `state`; LLM scripted to `save_memory` then answer; assert memory is in `MockStore` AND hook saw the post-extraction state
- [x] `tests/e2e/test_commands_e2e.py`
  - [x] `test_slash_compact_short_circuits_without_calling_llm` — zero-interaction scripted LLM; user input `/compact`; assert no `ReplayMismatchError` (LLM never called), dispatcher output reached state as an `AIMessage`. Guards the CommandMiddleware double-response bug from 0e21c21.
- [x] `tests/e2e/test_plugins_e2e.py`
  - [x] `test_plugin_tool_and_section_reach_running_graph` — `PluginContribution` with one tool (`ping() -> "pong-from-plugin"`) and one section (distinctive marker). Uses `capturing_scripted_llm` to inspect the system prompt the LLM received. Asserts (1) plugin section content reached the prompt, (2) `extensions` condition auto-activated, (3) plugin tool actually executed and returned its marker output. Surfaced: plugin tools' LLM-facing name is derived from `fn.__name__`, not `ToolCapability.name` — so plugin contributors need to name their function to match the intended tool name.
- [ ] **Regression guard verification:** revert commit `1383151` locally; `uv run pytest tests/e2e/test_deferred_tools_e2e.py` fails on `test_empty_deferred_registry_does_not_push_llm_toward_tool_search`; restore; passes. This proves the layer catches the class of bug it was built for. *(deferred — optional, can be done any time.)*

### Phase 4 — exhaustive feature audit + near-full coverage

Long-running. One PR per cluster. `tests/e2e/FEATURE_INVENTORY.md` is the per-feature tracker for this phase; this roadmap only tracks cluster-level progress.

- [x] Seed `tests/e2e/FEATURE_INVENTORY.md` — ~55 features catalogued across all 10 clusters with status=`inventoried` (or `covered` where Phase 3 already did the work). Cross-cluster meta-invariants section added for the "bug class" generalizations.
- [x] **Cluster A — Tool system.** `ToolRegistry`, `ToolCapability`, every standard tool (memory, retrieval, search, skill, async, UI, HITL), `DeferredToolRegistry`, `tool_search`, `call_deferred_tool`. Covered: memory (CRUD e2e + invalid-arg recovery), skills (list/read + unknown name), UI tools (all 4), deferred tools (full flow), async tasks (4 recoverable-error paths against empty worker registry), HITL (accept/response/ignore pause→resume round-trip), retrieve_result (happy-path + unknown key + offset/limit pagination). *Out-of-scope:* MCP adapters (needs in-process MCP server).
- [x] **Cluster B — Middleware stack.** All 12 middlewares individually + stack-ordering invariants. Covered: CommandMiddleware, ToolLoopGuardMiddleware, ToolErrorMiddleware, ResultPersistenceMiddleware, EmptyTurnMiddleware, StopHooksMiddleware (happy + blocking/non-blocking error paths), save_memory→hook ordering, QueuedInputMiddleware (APPEND/INTERRUPT/empty), PressureMiddleware (microcompact under load, light-load passthrough), ExtractionMiddleware (positive-emit JSON→MemoryRecord), CompletionGuardMiddleware (premature completion challenge), RuntimeStateMiddleware (smoke), PostRunBackstopMiddleware (record shape).
- [x] **Cluster C — Commands + prompt assembly.** Dispatcher (short-circuit + unknown-command fall-through), every built-in command (/help, /memory, /compact, /tools, /skills, /status, **/context**), condition-gating invariant, **PromptComposer STABLE/VOLATILE priority ordering**, **context-provider failure isolation** (one provider raising does not corrupt others).
- [x] **Cluster D — Plugins.** Happy-path tool+section+extensions activation, section-only, tool-only, multi-plugin composition, empty-registry gating, **worker-only contribution dispatch**, **multi-plugin id-collision precedence** (later registration wins via `configure_tools`).
- [x] **Cluster E — Memory + persistence.** CRUD e2e + gap-filling unit coverage: update unknown id returns None, type-relocation rewrites namespaces, get without memory_type probes every namespace, list_all_scopes returns only populated scopes, search with memory_type filter restricts scan. *Out-of-scope:* live sqlite/Postgres adapters (needs infrastructure).
- [x] **Cluster F — Context management.** PressureMonitor unit coverage (existing) + real multi-turn microcompact via PressureMiddleware e2e.
- [x] **Cluster G — Resilience.** Loop guards (unit + e2e + concurrent-isolation), stop hooks error paths (blocking vs non-blocking), backstop record shape, **tool-error retry semantics (retryable TimeoutError vs non-retryable ValueError observed in message stream)**.
- [x] **Cluster H — Builders + recursion.** build_deep_agent, build_reference_deep_agent, build_basic_deep_agent, **build_coding_agent**, recursion limit (build + runtime override + exhaustion), DEFAULT_RECURSION_LIMIT contract.
- [x] **Cluster I — Replay + streaming.** RecordedChatModel (Phase 3) + bind_tools unit + ConversationRecorder captures real-graph runs, save/load JSON round-trip, ReplayAssertions mismatch detection, recording drives a second graph run, ainvoke ↔ astream_events state parity, SSE wrapper emits tool_call_start/end + [DONE].
- [x] **Cluster J — Integrations.** FastAPI (invoke/list endpoints via TestClient + 404 path), A2A (Task envelope + Agent Card + aggregated card), supervisor (keyword-route dispatch with delegation record + no-agent graceful response), CLI scaffolder (generate + import + smoke invoke).
- [x] ≥90% overall line coverage (`uv run pytest --cov=langgraph_kit --cov-report=term-missing`) — **measured 90.55%** after the coverage-fill pass (23 new test files). Files still below 90% (`graphs/_builder.py`, `core/middleware/stop_hooks.py`, `core/persistent_memory.py`) have justification entries in `FEATURE_INVENTORY.md` — each remaining gap is an exception/defensive branch that is unreachable under supported configurations.
- [x] Every `ACTIVATION_SECTIONS` condition has a paired invariant test (section active ⇔ backing capability wired) — `memory`, `orchestration`, `skills`, `async_tasks`, `deferred_tools`, `extensions` all have invariants in `tests/e2e/test_invariants_e2e.py`.
- [x] Every middleware has ≥1 happy-path e2e + ≥1 empty/misconfigured edge-case test
- [x] Every `/command` has an e2e scenario invoking it end-to-end (all built-ins — `/help`, `/memory`, `/compact`, `/tools`, `/skills`, `/status`, `/context`)
- [x] All four builders exercised by ≥1 e2e scenario
- [x] **Regression guard verified.** Reverting `src/langgraph_kit/graphs/_builder.py` gating lines causes `test_empty_deferred_registry_does_not_push_llm_toward_tool_search` + `test_default_build_does_not_activate_deferred_tools_condition` to fail; restoring the fix returns them to green. The layer demonstrably catches the bug class it was built for.

Each cluster row expands inline into its own sub-checklist when it becomes the current focus. Keep collapsed until work on the cluster starts — avoid speculative planning.

---

## Resume instructions

1. Read the **Status** block at the top of this file.
2. Find the first unchecked (`- [ ]`) item in the active phase.
3. For Phase 4: open the relevant cluster's sub-checklist (expand the row inline if not yet expanded) and consult `tests/e2e/FEATURE_INVENTORY.md` for which feature rows are `inventoried` / `in progress`.
4. Open the most recent PR touching this file for context on any in-flight decisions or plan revisions.
5. When you finish a step: tick the box, update the **Status** block, and commit both changes together with the code change in the same PR.

## Tracker maintenance discipline

- **Tick boxes in the same PR as the work.** Ticking is not a separate commit.
- **Update the Status block on every phase or sub-step transition.** A stale status block is a bug — fix it in the same PR.
- **Don't delete ticked items.** They are the completion record.
- **If the plan changes** (new sub-step, reordered cluster, dropped feature), update this roadmap FIRST, in its own PR, then do the work. This file is the source of truth.
- **Surface new bug classes** in `FEATURE_INVENTORY.md`'s "Edge cases" column — not in this roadmap — and open issues for anything actionable.
