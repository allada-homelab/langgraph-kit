# Feature Inventory

Per-feature tracker for the Phase 4 exhaustive audit (see
[`TESTING_ROADMAP.md`](../../TESTING_ROADMAP.md)). Every row represents
one feature in langgraph-kit; the per-feature workflow is in
`TESTING_ROADMAP.md` §4c. This file is **the source of truth for what
is left to do within Phase 4** — the roadmap tracks cluster-level
progress only.

## Legend

- **Status:** `inventoried` (listed, not yet audited) · `in progress`
  (audit underway) · `covered` (all gaps closed) · `out-of-scope
  (justified)` (intentionally not e2e-testable — includes justification)
- **Columns:** Source / Purpose / Use cases / Edge cases / Existing
  coverage (unit, e2e) / Gaps / Status.

## Cluster A — Tool system

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `ToolRegistry` | `core/tools/registry.py` | Register + compile active tools bound to the LLM. | (1) Standard tools register at build; (2) `configure_tools` callback overrides; (3) Plugin contributions merge before callback. | Id collision; empty registry; `compile_tools()` called before registers. | `test_tools.py`, `test_registry.py` | None direct; exercised via every e2e. | Empty-registry behavior explicit. | inventoried |
| `ToolCapability` | `core/tools/capability.py` | Metadata wrapper around an async tool callable (id/name/description/tags/risk/fn). | (1) Wrap function + metadata; (2) Serialize to prompt hint; (3) Risk-aware dispatch (future). | `fn.__name__` vs `name` divergence (bit Phase 3.4 — LLM sees `fn.__name__`); no-arg tools; tools with `**kwargs`. | `test_r1_features.py` (indirect) | Phase 3.1 tests use it. | Document `name` vs `fn.__name__` in source docstring. | inventoried |
| `DeferredToolRegistry` | `core/tools/deferred.py` | Tools discoverable via `tool_search` but not bound at build. | (1) Populate via `configure_deferred_tools`; (2) Search → call_deferred_tool flow; (3) Empty registry auto-gates prompt. | Empty search result; unknown tool_id; stringified-JSON arguments; async tool bodies. | `test_r1_features.py` | `test_deferred_tools_e2e.py` (3 tests). | Parallel tool_search calls (same run, different queries). | covered |
| `tool_search` | `core/tools/deferred.py:128-169` | Keyword-match discovery over the deferred registry. | (1) LLM calls with a query; (2) Results formatted with ids + signatures; (3) Advisory appended after loop threshold. | No results (empty registry); ambiguous query matching; unicode content. | `test_r1_features.py` | `test_deferred_tools_e2e.py` (all 3 tests). | — | covered |
| `call_deferred_tool` | `core/tools/deferred.py:172-246` | Dispatcher for deferred tools; invoked by LLM. | (1) Happy-path invocation; (2) Sync + async fns; (3) Wrong args fall through with error. | Unknown tool_id; JSON-string arguments; wrong-shape arguments; missing arguments. | `test_r1_features.py` | `test_deferred_tools_e2e.py::test_deferred_tool_discoverable_and_callable`. | Wrong-shape arguments at e2e level (LLM emits bad args); concurrent dispatch. | inventoried |
| `save_memory` tool | `core/tools/memory_tools.py:22-61` | Persist a memory record. | (1) LLM decides to remember; (2) /memory command reads it back. | Invalid memory_type or scope (returns error string). | (indirect) | `test_middleware_ordering_e2e.py`, `test_tools_memory_e2e.py` (4 scenarios). | — | covered |
| `list_memories` tool | `core/tools/memory_tools.py:63-` | List stored memories by scope/type. | (1) User asks what's remembered; (2) Filter by type. | Empty store; invalid scope. | (indirect) | `test_tools_memory_e2e.py::test_save_then_list_memories`, `test_delete_memory_removes_it_from_list`. | Scope-filter explicit e2e. | covered |
| `search_memories` tool | `core/tools/memory_tools.py` | Semantic search over memories. | (1) Query-driven recall. | Empty result; non-matching query. | (indirect) | None. | Full flow e2e. | inventoried |
| `update_memory` / `delete_memory` tools | `core/tools/memory_tools.py` | CRUD mutations. | (1) Correct old facts; (2) Drop stale entries. | Unknown id; wrong scope. | (indirect) | `test_tools_memory_e2e.py::test_save_update_then_read_shows_update`, `test_delete_memory_removes_it_from_list`. | Unknown-id error path e2e. | covered |
| `retrieve_result` tool | `core/graph_builder/tools.py:86-94` | Pull persisted large tool outputs back into context on demand. | (1) Fetch a truncated prior result; (2) Dispatch from LLM. | Unknown key; expired. | (indirect) | `test_tools_retrieve_result_e2e.py` (full content + unknown key + offset/limit pagination). | — | covered |
| Skill tools (`list_skills`, `read_skill`) | `core/skills/tools.py` | Progressive-disclosure skill discovery. | (1) Discover available skills; (2) Load full instructions on demand. | No skills registered; unknown skill name (recoverable error). | (indirect) | `test_tools_skills_e2e.py` (2 scenarios: full flow + unknown-name error). | — | covered |
| Async task tools | `core/orchestration/async_tasks.py` | Fire-and-forget background agents. | (1) `start_async_task`; (2) `check_async_task`; (3) `cancel_async_task`; (4) `list_async_tasks`. | Unknown task id; cancellation races; completion after parent exit. | `test_integration_features.py` | `test_tools_async_e2e.py` (5 recoverable-error paths: empty worker registry, unknown id on check/cancel, invalid status filter, empty list). | Populated-registry happy path (needs wiring `available_graphs` at builder level). | covered |
| UI tools | `core/ui_events.py` + `core/artifacts.py` | `create_artifact`, `emit_progress`, `suggest_actions`, `add_citation`. | (1) Model emits artifact; (2) Progress pings UI; (3) Suggested next actions shown; (4) Citation added to source. | Malformed artifact payload; invalid progress counters (returns recoverable error); very large content. | `test_integration_features.py` | `test_tools_ui_e2e.py` (4 scenarios covering all four tools). | Very-large-content edge. | covered |
| HITL tools | `core/hitl/` | `approve_action` — request human approval before mutating operations. | (1) LLM gates a destructive action; (2) User approves / denies. | Denial path; timeout; pydantic reserved-name collisions (surfaced: `args` parameter clashed with StructuredTool's own `args` → renamed to `action_args`); GraphInterrupt must propagate past ToolErrorMiddleware (surfaced: the middleware was swallowing it as a permanent failure). | `test_integration_features.py` (indirect) | `test_tools_hitl_e2e.py` (accept / response / ignore pause→resume round-trip). | — | covered |
| MCP tool adapters | `core/plugins/mcp.py`, `core/plugins/mcp_client.py` | Wrap MCP-server tools as kit `ToolCapability`. | (1) Register MCP tools at build; (2) Dispatch to MCP server. | Server offline; schema drift. | `test_integration_features.py` | None. | Requires a stub MCP server — likely out-of-scope unless we host one in-process. | inventoried |

## Cluster B — Middleware stack

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `build_middleware_stack` | `core/graph_builder/middleware.py` | Assembles the 12-middleware chain in fixed order. | (1) Default build; (2) `tool_search_loop_threshold` override; (3) `stop_hooks` injection. | Missing memory_mgr; None pressure_monitor. | `test_loop_guard.py` | `test_deferred_tools_e2e.py`, `test_middleware_ordering_e2e.py` | Systematic: one happy-path + one empty-input test per middleware. | inventoried |
| `CommandMiddleware` | `core/commands/middleware.py` | Intercepts `/commands`, short-circuits via `jump_to: "end"`. | (1) Handled command; (2) Unrecognized command falls through; (3) Compacted-transcript variant. | Missing decorator (0e21c21 bug); non-string content. | `test_command_middleware.py` | `test_commands_e2e.py::test_slash_compact_short_circuits_without_calling_llm` | Unrecognized command falls through to LLM e2e; `/memory` e2e. | inventoried |
| `RuntimeStateMiddleware` | `core/resilience/runtime_state.py` | Initializes state bookkeeping. | (1) Fresh run init; (2) Retry tracking. | — | `test_reference_deep_agent.py` | `test_middleware_runtime_state_e2e.py` (run-completion smoke). | — | covered |
| `QueuedInputMiddleware` | `core/orchestration/` | Queued inputs from async subagents appear in the turn. | (1) Worker completes mid-conversation; (2) Queue empty. | Worker failure. | `test_queue.py` | `test_middleware_queued_input_e2e.py` (APPEND + INTERRUPT semantics + empty-queue no-op). | — | covered |
| `ToolErrorMiddleware` | `core/resilience/tool_error.py` | Converts tool exceptions into structured ToolMessage errors. | (1) Transient error retried; (2) Permanent error becomes `status="error"` ToolMessage with exception info; (3) Run never killed. | `max_retries=0`; exception types; retryable vs not. | `test_resilience.py` | `test_middleware_stack_e2e.py::test_tool_error_middleware_surfaces_structured_error`. | Retry-observed-twice e2e. | covered |
| `ToolLoopGuardMiddleware` | `core/resilience/loop_guard.py` | Soft-warn when a tool is called N+ times in a row. | (1) Default `tool_search` guard; (2) Custom tool+threshold; (3) Disabled via threshold=0. | Concurrent runs isolation; sibling-task accumulation (broken before Phase 3.1 fix). | `test_loop_guard.py` (incl. new concurrent-shape tests) | `test_deferred_tools_e2e.py::test_tool_loop_guard_advises_when_llm_spins_on_tool_search` | Guard on non-default tool via e2e. | covered |
| `PressureMiddleware` | `core/context_management/pressure_middleware.py` | Monitors window pressure, triggers compaction. | (1) Below threshold passthrough; (2) Above threshold compacts. | Monitor None; empty messages. | `test_context_management.py`, `test_pressure_full_compaction.py` | `test_middleware_pressure_e2e.py` (microcompact under load + light-load passthrough). | FULL_COMPACTION path (needs scripted compactor LLM). | covered |
| `ResultPersistenceMiddleware` | `core/context_management/result_persistence.py` | Persists large tool outputs and replaces with preview. | (1) Large content truncated; (2) Retrieve via `retrieve_result`. | Persist failure; retrieve on expired key. | `test_integration_features.py` | `test_middleware_stack_e2e.py::test_result_persistence_stores_large_output_and_trims_inline` (persist verified end-to-end; retrieve path still uncovered). | Retrieve round-trip e2e. | covered |
| `ExtractionMiddleware` | `core/memory/extraction_middleware.py` | Auto-extract memories from turn via an LLM worker. | (1) Agent didn't write memory itself; (2) Skip when agent already wrote; (3) Non-blocking on LLM failure. | LLM failure; empty messages. | `test_extraction.py`, `test_memory.py` | `test_middleware_extraction_e2e.py` (scripted JSON array → MemoryRecord reaches store under correct namespace with source=auto_extraction). | — | covered |
| `EmptyTurnMiddleware` | `core/resilience/empty_turn.py` | Prevents spinning on empty LLM output. | (1) Nudge up to N times; (2) Exit on final empty. | Per-turn nudge count. | `test_resilience.py` | `test_middleware_stack_e2e.py::test_empty_turn_middleware_does_not_spin_on_empty_output`. | Explicit nudge text in state. | covered |
| `CompletionGuardMiddleware` | `core/resilience/completion_guard.py` | Ensure minimum tool calls before completion. | (1) Min-1 default; (2) Allow direct-answer mode. | Threshold overrides. | `test_resilience.py` | `test_middleware_completion_guard_e2e.py` (premature-completion nudge injected into state). | — | covered |
| `StopHooksMiddleware` | `core/resilience/stop_hooks.py` | Run registered hooks after agent completes. | (1) Single hook; (2) Multiple hooks in order; (3) Non-blocking vs blocking. | Hook raises (blocking vs non-blocking). | `test_reference_deep_agent.py` | `test_middleware_ordering_e2e.py::test_save_memory_tool_persists_before_stop_hook_runs` | Blocking hook exception propagation e2e. | inventoried |
| `PostRunBackstopMiddleware` | `core/resilience/post_run.py` | Absolute backstop — records run metadata. | (1) Normal completion; (2) Error completion; (3) Interrupted run. | Metadata persist failure. | `test_resilience.py` | `test_middleware_runtime_state_e2e.py` (record shape: message_count, ai_messages, tool_calls, tool_errors, duration_seconds, completed_at, last_response_preview). | — | covered |

## Cluster C — Commands + prompt assembly

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `CommandDispatcher` | `core/commands/dispatch.py` | Route slash commands to handlers. | (1) Registered command; (2) Unknown command; (3) Handler with args. | Unrecognized command; empty args. | `test_command_middleware.py` | (indirect) | Unknown-command fall-through e2e. | inventoried |
| `/help` | `core/commands/builtins.py` | List available commands. | (1) List dump; (2) Specific command help. | — | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `/memory` | `core/commands/builtins.py` | List/search memories via command. | (1) Default scope; (2) Specific scope arg. | Empty store. | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `/compact` | `core/commands/builtins.py:82-126` | Compact transcript via `_microcompact`. | (1) Small state ("nothing to compact"); (2) Truncates large tool outputs. | Messages empty. | `test_command_middleware.py` | `test_commands_e2e.py::test_slash_compact_short_circuits_without_calling_llm`. | Large-transcript compaction observed in state. | inventoried |
| `/status` | `core/commands/builtins.py` | Summarize current context/state. | (1) Token counts; (2) Memory summary. | — | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `/tools [tag]` | `core/commands/builtins.py` | List tools, optionally filter by tag. | (1) All tools; (2) Tag filter. | Unknown tag. | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `/skills` | `core/commands/builtins.py` | List available skills. | (1) Default list. | No skills registered. | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `/context` | `core/commands/builtins.py` | Inspect the current prompt context. | (1) Dump composed prompt. | — | `test_command_middleware.py` | None. | Happy path e2e. | inventoried |
| `PromptComposer` | `core/prompt_assembly/composer.py` | Merge sections by priority + stability under active conditions. | (1) Base build; (2) Conditions gate sections; (3) Providers inject context. | Duplicate section ids; missing provider. | `test_prompt_assembly.py` | (indirect via every e2e). | Priority-conflict e2e. | inventoried |
| `SectionRegistry` | `core/prompt_assembly/sections.py` | Register prompt sections by id. | (1) Unique id register; (2) Id collision replaces. | Collision between core and plugin. | `test_prompt_assembly.py` | `test_plugins_e2e.py` (covers one collision path). | — | inventoried |
| `ACTIVATION_SECTIONS` | `core/prompt_assembly/activation.py` | Conditionally-included prompt fragments (`deferred_tools_awareness`, `skill_activation`, `extension_awareness`, `async_tasks`). | (1) Condition flips section on/off; (2) Auto-gating on capability. | Condition claimed with empty capability (the regression). | `test_r1_features.py`, `test_reference_deep_agent.py` | `test_deferred_tools_e2e.py::test_empty_deferred_registry_does_not_push_llm_toward_tool_search`. | **Invariant test for every condition/capability pair** (generalization of the deferred_tools fix). | inventoried |
| Context providers | `core/prompt_assembly/context_providers.py` | Inject thread/memory/tool context into the prompt. | (1) ThreadContext; (2) MemoryContext; (3) ToolContext; (4) Custom provider. | Provider raises. | `test_prompt_assembly.py` | (indirect) | Provider-failure isolation e2e. | inventoried |

## Cluster D — Plugins

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `PluginRegistry` | `core/plugins/registry.py` | Aggregate plugin contributions. | (1) Collect tools/sections/workers; (2) Empty registry no-ops. | Empty registry must NOT auto-activate `extensions`. | `test_reference_deep_agent.py` | `test_plugins_e2e.py`. | Worker dispatch e2e. | inventoried |
| `PluginContribution` | `core/plugins/registry.py` | Bundle tools+sections+workers per plugin. | (1) Tool-only plugin; (2) Section-only plugin; (3) Worker-only plugin. | Id collisions across plugins. | `test_reference_deep_agent.py` | `test_plugins_e2e.py::test_plugin_tool_and_section_reach_running_graph`. | Worker-only plugin e2e; multi-plugin collision. | inventoried |
| Plugin loader | `core/plugins/loader.py` | Entry-point discovery. | (1) Load from pkg metadata. | No plugins declared. | `test_integration_features.py` | None. | Load-from-pkg e2e (likely unit-sufficient). | inventoried |

## Cluster E — Memory + persistence

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `PersistentMemoryManager` | `core/memory/persistent.py` | CRUD + search facade over a LangGraph Store. | (1) create; (2) get; (3) update; (4) delete; (5) search; (6) list_by_scope. | Unknown id; wrong scope; store failure. | `test_memory.py` (+ new coverage: unknown-id update, type-relocation, cross-namespace get, list_all_scopes, search type-filter). | `test_tools_memory_e2e.py` | — | covered |
| Memory types and scopes | `core/memory/models.py` | Enums for memory classification. | (1) Valid value roundtrip; (2) Invalid rejected at tool entry. | Unknown enum values from LLM. | `test_memory.py` | None. | — | inventoried |
| `AutoMemoryExtractor` | `core/memory/extraction.py` | LLM-driven memory extraction from recent turns. | (1) Emit memory candidates; (2) Skip when agent wrote; (3) Swallow LLM errors. | Parse failures; rate limits. | `test_extraction.py` | `test_middleware_extraction_e2e.py` (positive emit). | — | covered |
| Store adapters | `persistence.py` | Wire LangGraph store from config (sqlite, postgres, in-memory). | (1) In-memory; (2) SQLite; (3) Postgres. | Missing DB; schema mismatch. | `test_memory.py` | None. | In-memory → sqlite e2e. | inventoried |

## Cluster F — Context management

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `PressureMonitor` | `core/context_management/pressure.py` | Compute signals about context window use. | (1) Fresh state low; (2) Long conversation high. | Empty messages. | `test_context_management.py` | `test_middleware_pressure_e2e.py` | — | covered |
| Compaction strategies | `core/context_management/compaction.py` | Microcompact, summarize, drop-prefix. | (1) Tool-result truncation; (2) Summary of older turns. | Nothing to compact. | `test_pressure_full_compaction.py` | `test_middleware_pressure_e2e.py` (MICROCOMPACT under real multi-turn load). | FULL_COMPACTION via running graph (scripted compactor LLM). | covered |
| Continuation | `core/context_management/continuation.py` | Hand-off between compacted state boundaries. | (1) Preserve task todos; (2) Resume mid-plan. | Empty todos. | `test_context_management.py` | None. | — | inventoried |

## Cluster G — Resilience

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| Loop guards | `core/resilience/loop_guard.py` | See Cluster B row. | — | — | — | — | Already tracked in Cluster B. | covered |
| Stop hooks | `core/resilience/stop_hooks.py` | Post-turn lifecycle callbacks. | (1) Single hook; (2) Multi-hook order; (3) Blocking vs non-blocking. | Hook exception; hook missing `on_turn_complete`. | `test_reference_deep_agent.py` | `test_middleware_ordering_e2e.py`. | Blocking/non-blocking behavior under error e2e. | inventoried |
| `PostRunBackstopMiddleware` | `core/resilience/post_run.py` | Absolute end-of-run cleanup. | (1) Metadata persisted; (2) Error path; (3) Empty-conversation safe. | Store write failure. | `test_resilience.py` | (indirect — visible as `run_metadata` namespace in Phase 3.2). | Metadata record shape assertion. | inventoried |
| Error recovery | `core/resilience/` | Retry + classification utilities. | (1) Transient retry; (2) Permanent propagate. | Unknown error types. | `test_resilience.py` | None. | Retry-observed e2e. | inventoried |

## Cluster H — Builders + recursion

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `build_deep_agent` | `graphs/_builder.py` | Low-level builder; exposes every knob. | (1) Core build; (2) `configure_tools` + `configure_deferred_tools`; (3) Plugins; (4) stop_hooks; (5) recursion_limit; (6) conditions override. | Empty deferred with explicit conditions (regression); missing llm. | `test_reference_deep_agent.py` | Every Phase 3 scenario. | Explicit-recursion-limit behavior; streaming mode. | inventoried |
| `build_reference_deep_agent` | `graphs/reference_deep_agent.py` | Curated full-stack overlay. | (1) Drop-in reference. | — | `test_reference_deep_agent.py` | Phase 1 spike (via build_reference_deep_agent) + `test_infrastructure_smoke.py` + `test_deferred_tools_e2e.py::test_empty_deferred_registry...`. | — | inventoried |
| `build_basic_deep_agent` | `graphs/basic_deep_agent.py` | Minimal agent — no kit middleware, memory, or tools. | (1) Raw model + prompt. | — | `test_reference_deep_agent.py` (indirect) | None. | Happy path e2e. | inventoried |
| `build_coding_agent` | `graphs/coding_agent.py` | Reference + coding workers + git context. | (1) File edit flow. | — | `test_coding_features.py` | `test_builders_e2e.py::test_build_coding_agent_smoke` (happy-path smoke). | — | covered |
| `DEFAULT_RECURSION_LIMIT` | `graphs/_builder.py:60-73` | Default supersteps per run. | (1) Default 100; (2) Per-build override; (3) Per-run override. | Hit the limit (raises GraphRecursionError); runtime override wins. | `test_reference_deep_agent.py` | None. | Build + runtime override e2e; GraphRecursionError on exhaust. | inventoried |

## Cluster I — Replay + streaming

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| `RecordedChatModel` | `replay/player.py` | LangChain chat model backed by a recording. | (1) Serve canned turns; (2) Support tool calls; (3) Bind tools as no-op. | Exhausted script; fuzzy match fallback. | `test_integration_features.py` + new `bind_tools` unit test. | Every Phase 3 e2e. | Fuzzy-match coverage. | covered |
| `CapturingScriptedChatModel` | `tests/e2e/helpers.py` | Test helper — record input messages. | (1) Prompt-inspection tests. | — | — (test-only) | `test_deferred_tools_e2e.py`, `test_plugins_e2e.py`. | — | covered |
| `ConversationRecorder` | `replay/recorder.py` | Capture real LLM traffic to a JSON file. | (1) Record-live; (2) Save/load. | Schema drift in `LLMInteraction`. | `test_integration_features.py` | `test_replay_recorder_e2e.py` (captures real-graph run + save/load round-trip + recording drives a second run). | — | covered |
| `ReplayAssertions` | `replay/assertions.py` | Structural assertions over recordings. | (1) Tool sequence match; (2) Message count match. | Drift between recording and live. | `test_integration_features.py` | `test_replay_recorder_e2e.py` (assert_same_tool_sequence / assert_tool_called / assert_final_output_contains + `assert_tool_sequence` helper). | — | covered |
| Streaming | `streaming.py` | `astream_events` helpers and filters. | (1) Token stream; (2) Internal-event suppression. | Back-pressure; aborted stream. | `test_streaming.py` | `test_streaming_parity_e2e.py` (ainvoke ↔ astream final-state parity; SSE wrapper emits tool_call_start → tool_call_end → [DONE]). | Token-level streaming (needs streaming LLM). | covered |

## Cluster J — Integrations

| Feature | Source | Purpose | Main use cases | Edge cases | Unit coverage | E2e coverage | Gaps | Status |
|---|---|---|---|---|---|---|---|---|
| FastAPI contrib | `contrib/fastapi.py` | Expose kit graphs as FastAPI routes. | (1) Mount a graph; (2) Request → ainvoke adapter. | Malformed request; streaming. | `test_contrib_fastapi.py` | `test_fastapi_e2e.py` (TestClient HTTP round-trip against /invoke + /agents list + 404 path). | create_app_lifespan full wiring (needs configure_from_settings). | covered |
| A2A protocol | `contrib/a2a.py` | Agent-to-agent request wrapping. | (1) Invoke; (2) Stream. | Invalid payload. | `test_integration_features.py` | `test_a2a_e2e.py` (invoke returns completed Task envelope + build_agent_card shape + aggregated card lists registered agents). | — | covered |
| MCP server contrib | `contrib/mcp_server.py` | Host kit graphs as an MCP server. | (1) Expose tool surface. | Handshake failures. | `test_integration_features.py` | None. | Requires in-process MCP client — likely out-of-scope unless we ship stubs. | inventoried |
| AGUI | `contrib/agui.py` | Agent-facing GUI events. | (1) Event stream; (2) Suggestion format. | Malformed event. | `test_feature_quality_pass3.py` | None. | — | inventoried |
| Supervisor pattern | `graphs/supervisor_agent.py` | Multi-agent supervisor. | (1) Dispatch to sub-agent; (2) Aggregate responses. | Sub-agent failure. | `test_integration_features.py` | `test_supervisor_e2e.py` (keyword-route dispatch + delegation record + no-agent graceful response). | — | covered |
| Echo agent | `graphs/echo_agent.py` | Reference "hello world" graph. | (1) Smoke the stack. | — | (indirect) | None. | Smoke e2e (optional — already covered by infra smoke test). | out-of-scope (covered by `tests/e2e/test_infrastructure_smoke.py`) |
| CLI scaffolder | `cli.py` | `python -m langgraph_kit.cli new <name>` — scaffolds new agent. | (1) Emit a working agent; (2) Emitted agent imports and invokes cleanly. | Template drift from kit API. | — | `test_cli_scaffolder_e2e.py` (scaffold + importable module check + smoke invoke via scripted LLM). | — | covered |

## Cross-cluster meta-invariant checklist

These aren't features per se — they're generalizations of the class
of bugs this whole effort was designed to catch. Check them off as
their invariant tests land.

- [ ] **Condition/capability pairing (generalized `deferred_tools` fix).** For every `ACTIVATION_SECTIONS` condition X: when X is active, the backing capability is actually wired. When the backing capability is empty, X is NOT active (unit) and its prompt section does NOT reach the LLM (e2e).
- [ ] **Middleware "no-op when input empty" coverage.** Every middleware in `build_middleware_stack` has an explicit empty-input test (stop_hooks=[], pressure_monitor with no messages, etc.) that asserts nothing is emitted, no exception is raised, and the graph still completes.
- [ ] **Tool name surfaces from `fn.__name__`.** Every plugin / deferred tool test asserts on the LLM-facing name, not the `ToolCapability.name` — surfaced in Phase 3.4.
- [ ] **Concurrent isolation.** Any shared-state primitive in the kit (streak counters, caches, pressure monitors) has at least one test running two agents concurrently and asserting their state stays separate.
- [ ] **Recursion + runtime override.** Build-time `recursion_limit` is honored at build; runtime `config={"recursion_limit": N}` overrides; exhaustion raises `GraphRecursionError` cleanly (no orphaned tasks, no silent truncation).
