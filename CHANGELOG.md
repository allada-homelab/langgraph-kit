# Changelog

All notable changes to this project are documented here. This project adheres to
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`RecordedChatModel.bind_tools` override.** The replay model
  subclasses `BaseChatModel`, and `BaseChatModel.bind_tools` raises
  `NotImplementedError` by default. `create_agent` and any LangChain
  agent flow call `bind_tools` during construction, so without an
  override the recorded model could not drive a real compiled graph
  — defeating the whole point of the replay system. The override is a
  pass-through (tool schemas don't change what a recording serves,
  since tool_calls are already baked into each
  `LLMInteraction.output_message`).

- **End-to-end test layer (`tests/e2e/`).** New pytest `e2e` marker,
  shared fixtures (`checkpointer`, `e2e_store`, `patched_build_llm`),
  and scripting helpers (`scripted_llm`, `tool_call_turn`, `answer`,
  `CapturingScriptedChatModel`, `assert_tool_invoked`,
  `last_ai_message`) for driving real compiled graphs with a scripted
  `RecordedChatModel`. **~87 scenarios across 30 files** now cover
  deferred tools, every standard tool (memory CRUD, skills, UI,
  async-task error paths, HITL approve_action pause/resume,
  `retrieve_result` pagination), every slash command, every middleware
  (including `QueuedInput`, `Pressure` microcompact under load,
  `Extraction` positive-emit, `CompletionGuard` premature-completion
  nudge, `PostRunBackstop` record shape, `RuntimeState` smoke),
  plugin contributions (section-only / tool-only / multi-plugin /
  empty-registry gating), stop-hook error paths, recursion-limit
  behavior, all four builders (`build_deep_agent`,
  `build_reference_deep_agent`, `build_basic_deep_agent`,
  `build_coding_agent`), replay (recorder captures real runs, save/load
  round-trip, ReplayAssertions mismatch detection, recording drives
  a second graph), streaming ↔ ainvoke parity, FastAPI HTTP
  round-trip, A2A Task envelope + Agent Card, supervisor keyword
  routing with delegation record, CLI scaffolder import + smoke
  invoke, and cross-cluster condition/capability invariants that
  generalize the `deferred_tools` bug class. See `TESTING_ROADMAP.md`
  and `tests/e2e/FEATURE_INVENTORY.md`.

### Fixed

- **`approve_action` parameter `args` renamed to `action_args`.**
  LangChain's `StructuredTool` reserves the name `args` on its
  internal schema; a tool function with a literal `args` parameter
  made LangChain mangle it to `v__args` at dispatch time, and the
  actual call then failed with
  `TypeError: approve_action() got an unexpected keyword argument 'v__args'`.
  The interrupt payload still speaks
  `{"action_request": {"args": ...}}` so the frontend / `/resume`
  contract is unchanged — only the LLM-visible parameter name on the
  tool signature changed. Surfaced by the HITL e2e tests in
  `tests/e2e/test_tools_hitl_e2e.py`.

- **`ToolErrorMiddleware` now re-raises `GraphInterrupt`.** The
  middleware previously caught every exception and converted it into
  a structured error `ToolMessage`, including LangGraph's
  `GraphInterrupt` control-flow signal. That turned an HITL pause
  into an error (the run continued as if the tool had failed, and
  the user's `Command(resume=…)` payload never reached the paused
  tool). The middleware now re-raises `GraphInterrupt` before its
  general `except Exception:` branch, so `interrupt()` calls from
  `approve_action` (and any future HITL tool) properly pause the
  graph. Surfaced by the same HITL e2e tests.

- **`ToolLoopGuardMiddleware` streak counter now survives real graph
  execution.** The guard previously stored per-run streak counts in a
  `ContextVar[dict]` via `.set(new_dict)`. `ContextVar.set` is
  copy-on-write per asyncio task, and LangGraph schedules each tool
  call as its own task — so every call saw `count=1` and the guard
  never fired under real execution. Unit tests missed the bug because
  they ran every call in a single coroutine. The guard now keys its
  streak counter on `thread_id` via a module-level dict
  (`runtime.execution_info.thread_id` in `abefore_agent` /
  `aafter_agent`, `runtime.config["configurable"]["thread_id"]` in
  `awrap_tool_call`), which is stable across all hook points within
  one `ainvoke` and different between concurrent threads. Two new unit
  tests (`test_streak_persists_across_sibling_asyncio_tasks`,
  `test_streak_isolated_between_concurrent_runs`) guard against
  regressions of the task-scheduling shape.

- **`deferred_tools` activation is now gated on registry population.**
  The `deferred_tools_awareness` prompt section tells the LLM to call
  `tool_search` to discover capabilities that aren't in its tool
  surface — useful when a `DeferredToolRegistry` is populated, actively
  harmful when it isn't (the LLM searches, finds nothing, and on
  recursion-bound runs can spin on `tool_search`). The builder now
  only auto-activates the `"deferred_tools"` condition when
  `bool(deferred_registry)` is true, and, for callers who pass an
  explicit `conditions=` set that includes `"deferred_tools"` without
  populating the registry, logs a warning and drops the condition.
  `DeferredToolRegistry` gained `__bool__` / `__len__` so callers and
  tests can cheaply check population. The generated-agent scaffold
  (`python -m langgraph_kit.cli new …`) and the reference
  `build_deep_agent` auto-defaults were updated to stop including
  `"deferred_tools"` unconditionally.

### Changed

- **Deep agents now default to `recursion_limit=100`** (up from LangGraph's
  native 25). `build_deep_agent`, `build_basic_deep_agent`,
  `build_reference_deep_agent`, and `build_coding_agent` all bind the
  default via `graph.with_config({"recursion_limit": 100})` so Pregel
  supersteps spent on prompt assembly, middleware, worker round-trips,
  and tool loops don't raise `GraphRecursionError` on realistic runs.
  The value is exposed as
  `langgraph_kit.graphs.DEFAULT_RECURSION_LIMIT` and can be overridden
  per-build via a new `recursion_limit=<n>` kwarg on every deep-agent
  builder, or per-run via `config={"recursion_limit": <n>}` on
  `ainvoke` / `astream_events` (runtime config wins over the build-time
  default). See the prominent call-out in `README.md`,
  `docs/agents/overview.md`, and the individual agent pages.

## [0.9.5] — 2026-04-22

### Fixed

- **Deferred tools are now actually callable end-to-end.** Previously
  `tool_search` was registered but no population path existed
  (`register_standard_tools` discarded the `DeferredToolRegistry` that
  `register_search_tool` returned), and even when a tool reached the
  registry there was no way for the LLM to invoke it — LangChain's
  tool-calling surface is bound at `create_agent` construction time and
  cannot pick up new tools mid-run. Fixes:
  - Added `build_call_deferred_tool(deferred)` — a dispatcher registered
    alongside `tool_search` on the active tool surface. The LLM calls
    `call_deferred_tool(tool_id=..., arguments={...})` and the
    dispatcher looks up the capability and invokes it. Handles sync and
    async callables, stringified-JSON arguments, wrong-shape arguments,
    and unknown ids — all returned as error strings so the model stays
    in control. Tool remains in the registry after invocation so it can
    be called repeatedly.
  - Rewrote `tool_search` output so the guidance actually resolves:
    surfaces each hit's `id` (rendered distinctly from the display name
    to avoid conflation), a one-line parameter signature via
    `inspect.signature`, and explicit instructions to use
    `call_deferred_tool` for invocation.
  - `register_standard_tools` now returns the `DeferredToolRegistry`
    instead of throwing it away, and `build_deep_agent` exposes a
    `configure_deferred_tools: (DeferredToolRegistry) -> None` callback
    mirroring the existing `configure_tools` hook, giving callers a way
    to populate the registry.

## [0.9.4] — 2026-04-22

### Fixed

- **Memory extractor no longer leaks JSON candidates into the user-facing
  chat stream.** `AutoMemoryExtractor`, `MemoryConsolidator`,
  `PressureMiddleware._full_compaction`, and `LLMRoutingStrategy.route` now
  tag their internal `llm.ainvoke(...)` calls with
  `langgraph_kit:internal` and a call-site-specific tag so consumers can
  filter the resulting `on_chat_model_stream` events out of
  `astream_events(version="v2")`. Previously, the extractor's JSON array
  (or `[]`) was indistinguishable from the main agent's reply and was
  appearing as trailing text in chat bubbles after the real response
  finished. The kit's `stream_agent_events` SSE helper now applies this
  filter automatically.
- **Memory extractor no longer crash-logs on unknown `type` values.** The
  extractor prompt gave the model enough latitude to invent enum members
  (observed: `"type": "assistant"`), which caused
  `MemoryType(candidate["type"])` to raise `ValueError` inside a
  `logger.exception` handler — producing a traceback that looked like a
  hard crash. Candidates are now pre-validated via a new
  `coerce_memory_type` helper and bad ones are dropped at WARN with the
  offending value named; sibling candidates still persist. Same guard
  applied to `MemoryConsolidator`'s merge action, which previously would
  have deleted the source records before crashing on the invalid enum.

### Added

- `langgraph_kit.core.internal_tags` module exposing `INTERNAL_TAG`,
  per-call tags (`MEMORY_EXTRACTION_TAG`, `MEMORY_CONSOLIDATION_TAG`,
  `CONTEXT_COMPACTION_TAG`, `AGENT_ROUTING_TAG`), and an
  `internal_llm_config(...)` helper. Consumers streaming events directly
  can filter any kit-internal call with
  `if INTERNAL_TAG in (event.get("tags") or ()): continue`.
- `langgraph_kit.core.memory.models.coerce_memory_type` for safe
  validation of LLM-produced memory type values.

## [0.9.0] — 2026-04-22

### Fixed

- `langgraph_kit.contrib.fastapi.create_agent_router` no longer breaks
  `FastAPI.openapi()` when routes reference the caller-supplied
  `CurrentUser` alias. The module previously used
  `from __future__ import annotations`, which stringified every route
  annotation and left `typing.get_type_hints()` unable to resolve the
  factory-local `CurrentUser` ForwardRef — producing
  `PydanticUserError: TypeAdapter[...] is not fully defined`. Dropped the
  future import on that module so annotations are evaluated at definition
  time and captured as live `Annotated` objects via the factory's closure.
  Regression test added in `tests/test_contrib_fastapi.py`.

### Added

- Release engineering overhaul ported from `arr-py-client`: full PR CI workflow
  (ruff + basedpyright + pytest matrix across Python 3.11–3.13 + coverage gate),
  tag-driven release workflow with CHANGELOG notes extraction + artifact upload
  + post-release version bump on `main`, mkdocs build + GitHub Pages deploy,
  pre-commit auto-format workflow, merge-conflict label workflow, and PR
  labeler with enforced category labels.
- `justfile` with standard recipes (`install`, `test`, `lint`, `fmt`,
  `typecheck`, `pre-commit`, `build`) for a consistent local dev loop.
- `.pre-commit-config.yaml` running ruff, codespell, and basic hygiene hooks.
- `mkdocs.yml` wiring the existing `docs/` tree for rendering.
- `CONTRIBUTING.md`.
- Dynamic version sourced from `src/langgraph_kit/__version__.py` via
  `hatch.version` — single source of truth, kept in sync by the release
  workflow.
- Strict pytest configuration (`filterwarnings = ["error"]`,
  `xfail_strict = true`, `--strict-markers`, `--strict-config`) and coverage
  reporting with a starting floor.

### Changed

- **Python support widened from `==3.13.*` to `>=3.11,<3.14`.** CI now tests
  3.11, 3.12, and 3.13. Ruff `target-version` set to `py311` so
  `keep-runtime-typing` works correctly across the supported range. 3.14 is
  held back pending upstream `langchain_core` dropping its `pydantic.v1`
  imports, which are incompatible with Python 3.14.

## [0.5.0] — 2026-04-10

Ninth feature-quality improvement pass (via the `improver` automation):
HITL formatting and UI validation tests, sliding-window EMA DR detection in
the continuation tracker, LLM-judge scoring clamp, metric-detection fix
(metadata-only, phone pattern in safety), human-readable HITL response
parsing, replay assertions on `status` field with output-similarity check,
deduplicated `SessionNotebook` section-finding, and ignoring the `.improver/`
state directory.

## [0.4.0] — 2026-04-10

Feature-quality improvements pass 008: compaction, consolidation, routing,
and guard refinements.

## [0.3.0] — 2026-04-10

Feature-quality improvements pass 007: prompt polishing, cost-table
accuracy, and secret-detection hardening.

## [0.2.0] — 2026-04-10

Early refinement passes (005–006): deleted the dead pruning module, trimmed
unused exports, added the first batch of registry module tests, and added
22 tests spanning streaming, cost models, and mermaid tracing.

## [0.1.0] — 2026-04-10

Initial public release of the LangGraph agent toolkit.

### Features

- **Core agent config** (`AgentConfig`, `configure`, `get_config`) with
  multi-provider LLM factory (`build_llm`) covering OpenAI, Anthropic, and
  Google.
- **Persistent memory system** with multi-scope support (personal, team,
  worker) via LangGraph `Store`, plus thread-local `SessionNotebook` and
  LLM-powered post-turn extraction middleware.
- **Tool capability model** (`ToolCapability`, `ToolRisk`) with registry,
  deferred/lazy discovery, and memory-CRUD tools for agents.
- **Slash-command dispatch** with middleware interception (`/help`,
  `/memory`, `/context`, `/compact`, `/status`, `/tools`, `/skills`).
- **Context management**: token-budget monitoring, conversation compaction
  (full/partial), and continuation tracking with DR detection.
- **Prompt assembly**: layered composition with caching, section activation
  rules, dynamic runtime context providers, and git integration.
- **Multi-agent orchestration**: supervisor/coordinator patterns, async
  fire-and-forget workers, store-backed per-thread message queue.
- **Resilience**: premature-completion guards, empty-turn nudging,
  structured tool-error handling with transient retry.
- **Human-in-the-Loop**: interrupt-based approval (`ActionRequest`,
  `HumanInterrupt`, `HumanResponse`) and `approve_action` tool.
- **Streaming**: SSE events with `astream_events` v2 for rich UI events
  and artifacts.
- **Evaluation framework**: rule-based and model-graded metrics with
  prompt templates for faithfulness, helpfulness, safety, task completion,
  and tool efficiency.
- **Replay/recording** for deterministic testing of recorded traces.
- **Plugins + MCP**: plugin registry, MCP client factory, and an MCP server
  factory in `contrib`.
- **Integrations** (`contrib/`): FastAPI router factory, AGUI protocol,
  A2A SDK, MCP server.
- **Graph definitions**: `echo_agent`, `r0_agent`, `deep_agent`,
  `coding_agent`, `supervisor_agent`.
