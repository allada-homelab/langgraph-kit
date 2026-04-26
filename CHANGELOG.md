# Changelog

All notable changes to this project are documented here. This project adheres to
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Prompt versioning + run-level version tagging.** `PromptSection`
  gains a free-form `version: str` field (default `"1"`,
  caller-controlled — not a content hash) and `SectionRegistry` now
  keeps a per-id version map with explicit `set_current` /
  `list_versions` / `current_versions` APIs. Re-registering the same
  id with a new version auto-promotes it; pass `set_current=False` to
  stage a candidate without going live. `get_active()` still returns
  one section per id (the current one), so existing graph code
  doesn't change. `build_agent_run_config` accepts an optional
  `prompt_versions` mapping that's surfaced as
  `metadata["prompt_versions"]` on the run — visible per-trace in
  Langfuse for cohort analysis of a rollout. The A/B router and
  persistence story (rollout strategies, multi-worker coordination)
  are intentionally deferred — this PR ships the primitives #18
  asked for, the higher-level orchestration is out of scope here.
  Closes part of [#18](https://github.com/allada-homelab/langgraph-kit/issues/18).

- **Partial replay + recording overrides.** `ReplayRunner.run()` now
  accepts `start_at` / `stop_at` (Python slice semantics, negative
  indices supported) to replay a sub-range of recorded user turns —
  useful for debugging "show me what happens after turn 5" without
  rerunning the prefix verbatim. New `RecordingOverrides` model lets
  callers swap an LLM `output_message` at specific call ordinals
  without mutating the recording on disk, supporting "what if the
  model had said X here?" trajectory experiments. Both knobs are
  optional; default behavior is unchanged. `step_mode` (yielding
  pause-points between turns) is intentionally deferred — separate
  API surface, easy to add later. Fixes
  [#15](https://github.com/allada-homelab/langgraph-kit/issues/15).

- **Dev-mode hot-reload primitive.** New `langgraph_kit.dev.Reloader`
  watches a list of paths via stdlib mtime polling and fires a
  callback (sync or async) on each batch of changes. Default ignore
  list keeps `__pycache__` / `.pyc` / `.venv` / VCS directories out
  of the watch surface. No third-party dependency on
  `watchfiles` / `watchdog`. The full `langgraph-kit dev` server
  (file-watch + agent rebuild + checkpoint preservation + inspector
  UI) is multi-PR effort; this lands the foundation. Fixes
  [#36](https://github.com/allada-homelab/langgraph-kit/issues/36).

- **Disaster-recovery export / import.** New
  `langgraph_kit.core.dr.DisasterRecoveryManager` exports Store
  contents as JSON Lines (manifest header first, one record per line)
  and re-imports the same shape back. Three import modes — `replace`
  (clear each namespace before importing, the loud default),
  `append` (skip existing keys), `merge` (last value wins). Manifest
  carries `schema_version` (currently 1) so future format changes
  reject old files cleanly. Not a substitute for full database
  backups — a complement for selective restore (per-tenant,
  per-thread, etc.). The HTTP admin endpoint and the CLI integration
  are deferred to a follow-up. Fixes
  [#35](https://github.com/allada-homelab/langgraph-kit/issues/35).

- **Schema versioning + lazy forward migrations.** New
  `langgraph_kit.core.migrations` module ships a `Versioned` Pydantic
  mixin (adds `model_version: int`), a `Migration` (forward step
  ``v → v+1`` operating on dicts pre-validation), a `MigrationRegistry`
  keyed by `(model_class, source_version)`, and a `migrate_dict`
  helper that walks the registry until a payload reaches the model's
  current `MODEL_VERSION`. Forward-only by design (down-migrations are
  easy to lose data on); legacy rows without a `model_version` field
  are treated as v1 and walked forward; missing steps raise
  `MissingMigrationError` so deployments fail loud rather than
  silently misread old data. Adopting this for the kit's own models
  (`MemoryRecord`, `SessionNotebook`, `AsyncTask`, …) lands as those
  models bump their schemas. Fixes
  [#34](https://github.com/allada-homelab/langgraph-kit/issues/34).

- **Per-user data lifecycle: export / delete / anonymize.** New
  `langgraph_kit.core.lifecycle.DataLifecycleManager` exposes the
  GDPR-friendly trio over the LangGraph Store. Operates on the
  namespaces where `user_id` is already present today (the per-user
  thread index and the actor-keyed audit log); coverage will widen
  as #33 multi-tenancy adds tenant scoping to the rest. Anonymize
  uses a salted SHA-256 pseudonym (`anon-<16 hex>`) so records keep
  their analytical shape while the link to a real person is severed.
  Every lifecycle call writes a `DATA_EXPORT` / `DATA_DELETE` audit
  entry — the record of "user X requested deletion at time T" lives
  on after the data itself is gone. Fixes
  [#31](https://github.com/allada-homelab/langgraph-kit/issues/31).

- **Prometheus-format metrics primitives + `/metrics` ASGI endpoint.**
  New `langgraph_kit.observability_metrics` module ships pure-Python
  `Counter`, `Gauge`, `Histogram` (cumulative buckets, default
  spans 5 ms to 30 s), a `MetricsRegistry`, and a `MetricsEndpoint`
  ASGI app that renders the registry as the standard text-exposition
  format. No third-party dependency added — the kit's metrics needs
  are bounded and importing `prometheus_client` would cost more than
  the ~200 LOC implementation. Wiring kit-internal counters (LLM
  tokens, tool calls, compactions, rate-limit hits, HITL interrupts)
  is deferred to a follow-up. Fixes
  [#28](https://github.com/allada-homelab/langgraph-kit/issues/28).

- **Per-user rate limiting.** New `langgraph_kit.contrib.rate_limit`
  module with a token-bucket primitive (`TokenBucket`), an
  in-memory backend (`InMemoryRateLimitBackend`), and an ASGI
  middleware (`RateLimitMiddleware`) that emits `429 Too Many
  Requests` with a `Retry-After` header on bucket exhaustion. Keyed
  by the FastAPI `current_user` dependency by default; anonymous
  traffic shares one bucket; health-check paths bypass. Backend is
  swappable via the `RateLimitBackend` Protocol so a multi-process
  Redis backend can drop in later (#27 cross-process consistency).
  Tokens-per-day enforcement (the second limit in the issue) is
  deferred — it integrates with the existing `BudgetManager`. Fixes
  [#25](https://github.com/allada-homelab/langgraph-kit/issues/25).

- **Append-only audit log.** New `langgraph_kit.core.audit` module ships
  `AuditAction` (bounded enum: agent_invoke, memory_create/update/delete,
  hitl_approve/reject, injection_detected, output_redacted, data_export/
  delete), `AuditEntry` (immutable five-tuple of timestamp/actor/action/
  target/metadata), and `AuditStore` (the only sanctioned read/write
  surface). Storage is time-bucketed by year-month under `("audit",
  "YYYY_MM")` so monthly listings stay cheap; `query(...)` walks
  buckets newest-first so the common "last N entries" path stops as
  soon as it has enough. Store write failures are logged and swallowed
  so audit never blocks a real action. The FastAPI admin endpoint that
  queries the log is deferred to a follow-up. Fixes
  [#24](https://github.com/allada-homelab/langgraph-kit/issues/24).

- **Outbound PII / secret redaction.** New `OutputSafetyMiddleware`
  scans the most recent `AIMessage` after every turn and rewrites
  matched substrings with `[REDACTED]` before the message reaches the
  user. Default mode `"redact"`; `"warn"` flags only via
  `additional_kwargs`; `"off"` disables. Configurable via
  `AgentConfig.output_safety_mode`. Patterns cover credentials (API
  keys, GitHub PATs, Bearer tokens, AWS / Slack / Stripe / GCP) and
  PII (email, US phone, SSN, credit-card with Luhn validation to
  suppress 16-digit-but-not-actually-a-card false positives).
  Helpers `scan_for_unsafe_output` and `redact` are exported for
  in-process callers. Per-turn idempotent (won't re-redact the
  placeholder). Fixes
  [#23](https://github.com/allada-homelab/langgraph-kit/issues/23).

- **Inbound prompt-injection scanner.** New `langgraph_kit.core.security`
  module ships `INJECTION_PATTERNS` (regex set covering known phrasings
  like "ignore previous instructions", persona override, jailbreak
  vocabulary, system-prompt exfil, pseudo-system tags) and a
  `PromptInjectionGuardMiddleware` that scans the most recent
  `HumanMessage` on every turn. Default mode `"warn"` logs at WARNING
  and tags `additional_kwargs` with the matched pattern names so audit /
  observability layers can react; `"off"` disables. An optional
  `classifier=` hook handles paraphrases (regex stays the fast-path).
  Wired into the standard middleware stack via
  `AgentConfig.prompt_injection_mode`. The richer `"quarantine"` mode
  (narrow tool surface for the affected turn) is deferred to a
  follow-up — it depends on the audit-log infrastructure tracked in
  #24. Fixes
  [#22](https://github.com/allada-homelab/langgraph-kit/issues/22).

- **RAG primitives — foundation layer.** New `langgraph_kit.core.rag`
  module: `Document`, default `word_chunker` (3.2k chars / 200-char
  overlap, no mid-word splits), `RetrievalIndex` with
  ingest/search/delete on top of any LangGraph `Store`, and
  `build_search_knowledge_tool(index)` factory that wraps the index as
  an agent-facing `search_knowledge` tool. Embedding function is
  caller-supplied (same convention as semantic memory search from #8)
  — no new heavyweight dependencies. Cosine helper extracted to
  `langgraph_kit.core._vector_math` for reuse across memory and RAG.
  Citation verification + grounding-eval rubric are deferred to a
  follow-up. Fixes
  [#16](https://github.com/allada-homelab/langgraph-kit/issues/16).

- **CI-friendly exits for `python -m langgraph_kit.evals`.** Four new
  flags: `--fail-under N` exits non-zero when the overall pass rate is
  below N; `--baseline FILE` compares against a stored slim JSON
  report and fails on regression; `--baseline-tolerance` allows a
  configurable drop; `--ci-json PATH` writes a slim, schema-versioned
  JSON document suitable for CI artifact upload and as input to a
  future `--baseline` run. Helpers `compute_overall_pass_rate`,
  `report_to_ci_json`, and `check_ci_thresholds` are exported from
  `langgraph_kit.evals.report` for in-process use. Fixes
  [#14](https://github.com/allada-homelab/langgraph-kit/issues/14).

- **SSE heartbeats and event ids in `stream_agent_events`.** Every emitted
  chunk now carries an `id: <n>` line with a per-stream monotonically-
  increasing sequence number, and a configurable
  `heartbeat_interval` (default 15 s) emits
  `{"heartbeat": {"ts": ..., "last_event_id": ...}}` chunks during quiet
  periods so proxies / load balancers don't drop idle connections. Set
  `heartbeat_interval=None` to disable. Existing `error` events continue
  to fire before `[DONE]`. The durable replay log + `/stream/resume`
  endpoint that consume the `id:` line are deferred to a follow-up — for
  now the id makes the contract forward-compatible. Fixes
  [#11](https://github.com/allada-homelab/langgraph-kit/issues/11).

- **Wire up four `ToolCapability` orphan fields.** Three previously-advisory
  fields are now enforced by bundled middleware (Fixes
  [#6](https://github.com/allada-homelab/langgraph-kit/issues/6)):
  - `max_output_chars` (new field) — per-tool override of
    `ResultPersistenceMiddleware`'s threshold so chatty tools (`read_file`)
    can declare their own caps. The legacy `max_output_tokens` is kept as
    an advisory hint for caller-supplied tokenizer-aware middleware.
  - `offload_large_results` — `ResultPersistenceMiddleware` now consults
    the flag; setting `False` opts the tool out of persistence regardless
    of size. **Default flipped from `False` to `True`** (see Changed).
  - `interrupt_before` — new `AutoInterruptMiddleware` (in
    `core.hitl.auto_interrupt`) auto-pauses for HITL approval before any
    tool whose capability declares this flag. Coexists with the manual
    `approve_action` tool: use `interrupt_before` for tools whose risk is
    intrinsic, `approve_action` when the agent should decide whether to
    ask.
  - `coordinator=True` keyword on `build_deep_agent` wires `CoordinatorMode`
    (was previously unreachable from the public builder): narrows the tool
    surface to `ToolRisk.READ_ONLY`, merges coordinator prompt sections,
    and activates the `coordinator` / `orchestration` section conditions.
  - The `SkillMetadata.allowed_tools` re-add is deferred — it depends on
    `#7` (active skill state) and re-adding the field without that
    plumbing would re-introduce the same dead-contract problem.

### Changed

- **`ToolCapability.offload_large_results` default flipped from `False`
  to `True`** to match the expected meaning of the flag now that
  `ResultPersistenceMiddleware` honors it. Pre-flip the field was an
  ignored hint; callers who set `False` explicitly now get the opt-out
  behaviour they were trying to express. Callers who relied on the
  silent default see no behaviour change for over-threshold results
  (which were already being persisted unconditionally).

- **Opt-in structured-output validation via `StructuredOutputMiddleware`.**
  Pass `output_schema=` to `build_deep_agent` and the agent's terminal
  `AIMessage` is validated against your Pydantic schema. The model is
  asked to wrap its structured payload in a single
  `<output_schema>{...}</output_schema>` block (matches the existing
  `CompactionResult` convention); on mismatch the middleware retries
  with the schema rendered as JSON Schema, capped at 2 nudges. No
  provider-native `response_format` plumbing — provider-agnostic by
  design. Helpers `format_schema_instruction(schema)`,
  `extract_structured_output(content)`, and
  `parse_structured_output(content, schema)` are exported for prompt
  composition and post-run extraction. `AgentMetadata.output_schema`
  surfaces the contract for discovery. Fixes
  [#17](https://github.com/allada-homelab/langgraph-kit/issues/17). See
  [docs/resilience/structured-output.md](docs/resilience/structured-output.md).

- **Opt-in semantic search for `PersistentMemoryManager`.** Pass an async
  `embedding_fn` (or set `AgentConfig.memory_embedding_fn`) to index
  records on create/update and rank `search()` by cosine similarity
  instead of keyword overlap. No silent fallback — if no embedding
  function is configured the search uses case-insensitive token
  overlap against `title` / `summary` / `body`, which is deterministic
  regardless of `Store` backend support for `asearch(query=...)`. Fixes
  [#8](https://github.com/allada-homelab/langgraph-kit/issues/8). See
  [docs/memory/persistent-manager.md](docs/memory/persistent-manager.md#searchquery-str-scope-memoryscope-memory_type-memorytype--none--none-limit-int--5---listmemoryrecord).

- **Phase 4 testing complete — 90.55% line coverage, 765 tests passing.**
  Grew the test suite from a 442-test / ~72%-coverage baseline to 765
  tests / 90.55% coverage across 23 new module-level coverage files and
  7 new e2e scenario files. New module-level coverage: evals
  (`rule_based`, `model_graded`, `report`, `runner`), AG-UI streaming +
  encoder, MCP server + client manager (with mocked MCP modules), git
  worktree tools, async task manager, replay runner + extract helpers,
  observability Langfuse gate, persistence URL normalization + both
  connection branches, plugin loader, LLM factory provider routing,
  cost callback token accumulation, tracing storage CRUD, shared memory
  publish/sync + secret scanning, consolidation action application,
  agent memory manager CRUD. New e2e coverage: `search_memories` tool
  variants, deferred-tool argument-validation edges, prompt-composer
  STABLE/VOLATILE priority + provider-failure isolation, tool-error
  retry semantics (retryable `TimeoutError` vs non-retryable
  `ValueError` observed in message stream), streaming-mode
  `build_deep_agent`, `register_all` agent registration, and
  `skills`/`async_tasks`/`extensions` ACTIVATION_SECTIONS invariants.
  Regression guard verified — reverting the `deferred_tools` gating fix
  causes the dedicated e2e tests to fail; restoring it returns them to
  green.

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

### Security

- **`retrieve_result` now scopes persisted tool results per-thread.**
  `ResultPersistenceMiddleware` used to write large tool outputs under a
  flat `("tool_results",)` namespace, and `retrieve_result` looked them
  up there too — any agent that learned a ref hash could read any other
  thread's stored content. Both sides now use `("tool_results",
  thread_id)`, so refs are unreachable outside the thread that wrote
  them. Pre-existing refs written by earlier versions are unreadable
  after upgrade; re-running the workflow regenerates them.

- **FastAPI execution endpoints now verify thread ownership.** `stream`,
  `invoke`, `resume`, `resume/stream`, `fork`, `queue` (POST+GET),
  `messages`, `state`, and `history` previously accepted any
  authenticated user's request for any `thread_id`. A new
  `_verify_thread_owner` helper loads the `ThreadMetadata` record via
  `ThreadManager` and returns 404 on owner mismatch (matching the
  behaviour of the metadata endpoints, which already checked). 404 (not
  403) avoids leaking thread existence to probing users. The
  stream/invoke paths allow requests with an unclaimed `thread_id`
  through to `_ensure_thread`, which then claims ownership for the
  caller.

### Fixed

- **`a2a.py` `Request` import moved to module scope.** FastAPI's
  route-signature type-hint resolver runs against module globals, not
  the enclosing function's closure. With `from __future__ import
  annotations` in effect (so all annotations are stringified), a
  `from fastapi import Request` scoped inside `create_a2a_router()`
  left `Request` invisible to FastAPI at route-binding time. FastAPI
  silently fell back to treating `request: Request` route parameters
  as query params, and every A2A endpoint then 422'd at call time on
  any real request. The import now lives at module scope with a
  comment documenting why it must stay there and a `# noqa: TC002`
  because basedpyright would otherwise suggest moving it into
  `TYPE_CHECKING`. Surfaced by the new `test_a2a_e2e.py` scenarios
  that actually invoke the routes via `TestClient`.

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
