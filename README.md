# langgraph-kit

A batteries-included toolkit for building production-grade [LangGraph](https://github.com/langchain-ai/langgraph) agents — persistent memory, rich tool capabilities, multi-agent orchestration, context management, slash commands, HITL, and streaming SSE, all composable and all optional.

> **Status:** Alpha (v0.9.0). APIs may still evolve before 1.0.
> **License:** AGPL-3.0-or-later.
> **Python:** 3.11 – 3.13.

## Why langgraph-kit?

LangGraph gives you the graph primitives. **langgraph-kit gives you the rest of the stack** — the pieces every non-trivial agent ends up re-implementing: typed memory with auto-extraction, a tool registry with risk levels and HITL, a prompt composer tuned for Anthropic prompt caching, context-pressure middleware, a slash-command dispatcher, a ready-made FastAPI router, and a fully-wired reference agent you can clone.

Every subsystem is independent. Use just the memory manager if that's all you need — or wire everything together via the reference deep agent and start overlaying your domain on top.

## Features

- **Multi-provider LLM factory** — OpenAI, Anthropic (Claude), and Google (Gemini) via a single `build_llm()` call; provider auto-detected from model name.
- **Persistent memory** — typed (`USER` / `FEEDBACK` / `PROJECT` / `REFERENCE`) and scoped (`user` / `assistant` / `project` / `team`) records stored in LangGraph's `Store`, with agent-callable CRUD tools, semantic search, LLM-powered auto-extraction, and background consolidation.
- **Session notebook** — per-thread structured scratch-space (current state, task spec, workflow, errors, results) that survives compaction.
- **Tool registry with capability metadata** — every tool carries a risk level (`READ_ONLY` / `MUTATING` / `DESTRUCTIVE`), profile/worker/tag filters, and optional prompt-guidance fragments that are injected into the system prompt.
- **Prompt assembly with cache-aware ordering** — `STABLE` / `VOLATILE` / `CONDITIONAL` sections composed stable-first to maximize Anthropic prompt-cache hits, plus pluggable `ContextProvider`s for dynamic runtime context.
- **Context pressure management** — token estimation, microcompaction, full LLM-driven compaction, and a circuit breaker — applied automatically via middleware.
- **Resilience middleware** — completion guard (detects premature stops), empty-turn nudger, structured tool-error recovery, and a post-run backstop.
- **Slash-command dispatcher** — transport-independent `/help`, `/memory`, `/context`, `/compact`, `/status`, `/tools`, `/skills` built in; short-circuits the LLM entirely on command matches.
- **Multi-agent orchestration** — declarative worker definitions (`researcher` / `implementer` / `verifier`), fire-and-forget async tasks, a per-thread message queue for busy threads, and a read-only coordinator profile.
- **Human-in-the-loop** — `approve_action` tool and `interrupt_before=True` capabilities that pause the graph via LangGraph interrupts and stream approval requests to the frontend.
- **Rich UI events** — typed streaming events for artifacts (code, markdown, diagrams, tables), progress updates, suggestion chips, and citation cards, delivered alongside tokens over SSE.
- **Skills (progressive disclosure)** — agents discover and load `SKILL.md` files on demand instead of bloating the system prompt.
- **MCP integration** — wrap any [Model Context Protocol](https://modelcontextprotocol.io/) server's tools as native `ToolCapability` entries.
- **Plugin system** — drop `.py` files with a `contribute()` function into a plugins directory to extend the registry.
- **Ready-made FastAPI router** — 11 endpoints covering streaming, invoke, thread state, message queue, HITL resume, and checkpoint branching/forking.
- **Persistence out of the box** — `AsyncPostgresSaver` + `AsyncPostgresStore` for production, `AsyncSqliteSaver` + `InMemoryStore` for local dev, switched by the `database_url` scheme.
- **Observability** — first-class [Langfuse](https://langfuse.com/) tracing, run-config builder, and per-thread token budgets.
- **CLI scaffolding** — `python -m langgraph_kit.cli new <agent_id>` generates a complete agent template following the kit's conventions.
- **Evaluation framework** — `evals/` module with a runner, reports, and both rule-based and model-graded metrics.
- **Fully typed** — ships with `py.typed`, type-checked under `basedpyright`.

## The Reference Deep Agent

The toolkit ships with [`reference-deep-agent`](src/langgraph_kit/graphs/reference_deep_agent.py) — a full-stack general-purpose agent that wires every kit feature together. **Clone this agent as the starting point for any new domain-specific agent** (see [`coding_agent.py`](src/langgraph_kit/graphs/coding_agent.py) for the canonical extension pattern).

It is built on the `deepagents` framework and the shared [`build_deep_agent`](src/langgraph_kit/graphs/_builder.py) skeleton, so you get the entire feature stack below with one call:

```python
from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent

graph, dispatcher = build_reference_deep_agent(checkpointer, store, mcp_tools=[])
```

> **⚠️ Recursion limit defaults to `100`** — not LangGraph's native `25`. A full-stack deep agent easily burns through 25 supersteps on a single real task (middleware passes, worker round-trips, tool loops), so every deep agent built by this kit binds `recursion_limit=100` via `.with_config()`. Raise it for long autonomous runs — pass `recursion_limit=500` to any `build_*_deep_agent` / `build_coding_agent` call, or override per-run with `config={"recursion_limit": 500}` on `ainvoke` / `astream_events`. The constant lives at [`langgraph_kit.graphs.DEFAULT_RECURSION_LIMIT`](src/langgraph_kit/graphs/_builder.py).

### What's wired in

**Layered prompt assembly** — five core sections registered at build time:

| Section | Stability | Purpose |
|---|---|---|
| `core_identity` | `STABLE` | Agent identity and operating principles |
| `memory_instructions` | `CONDITIONAL` (memory) | How to use persistent memory responsibly |
| `orchestration_instructions` | `CONDITIONAL` (orchestration) | When and how to delegate to workers |
| `continuation_guidance` | `STABLE` | When to continue vs. stop on no-progress |
| `ui_interaction` | `STABLE` | How to use `emit_progress` / `suggest_actions` / `add_citation` / `approve_action` |

Stable sections are placed first to maximize Anthropic prompt-cache hits; volatile tool-guidance fragments and three default context providers (`Thread`, `Memory`, `Tool`) are appended per turn.

**Full 11-middleware stack** — applied in order by [`build_middleware_stack`](src/langgraph_kit/core/graph_builder/middleware.py):

1. `CommandMiddleware` — intercepts `/`-prefixed user messages and short-circuits the LLM on handled commands.
2. `RuntimeStateMiddleware` — populates per-turn runtime state available to other middleware.
3. `QueuedInputMiddleware` — drains the per-thread message queue at the start of each turn and injects buffered messages.
4. `ToolErrorMiddleware` — wraps tool calls, converts exceptions to structured errors the agent can reason about, and retries transient failures.
5. `PressureMiddleware` — estimates tokens and applies the selected mitigation strategy (`MICROCOMPACT`, `SESSION_ASSISTED`, `FULL_COMPACTION`, or `STOP` circuit breaker at 3× compaction failures).
6. `ResultPersistenceMiddleware` — offloads large tool outputs to the store to free up context.
7. `ExtractionMiddleware` — runs post-turn LLM-powered memory extraction, respecting the memory taxonomy (don't memorize what's already in the repo).
8. `EmptyTurnMiddleware` — nudges the model with a concrete instruction when it produces no output.
9. `CompletionGuardMiddleware` — detects premature completion heuristically and challenges the agent to justify stopping.
10. `StopHooksMiddleware` — runs registered stop hooks at graph end.
11. `PostRunBackstopMiddleware` — final safety check after graph execution.

**Worker (sub-agent) definitions** — pre-composed as [`GENERAL_WORKERS`](src/langgraph_kit/core/orchestration/workers.py):

| Worker | Role |
|---|---|
| `researcher` | Finds information, reads docs, searches code |
| `implementer` | Writes code, makes changes, builds features |
| `verifier` | Reviews changes, runs tests, validates output |

The primary agent delegates bounded work via `task`/`start_async_task` tools; each worker runs on its own thread with its own checkpointed state.

**Standard tool set** — registered by [`register_standard_tools`](src/langgraph_kit/core/graph_builder/tools.py):
- Five memory CRUD tools (`save_memory`, `list_memories`, `search_memories`, `update_memory`, `delete_memory`).
- UI event tools (`emit_progress`, `suggest_actions`, `add_citation`, `create_artifact`).
- HITL `approve_action` for destructive operations.
- Skills discovery (`discover_skills`, `get_skill_guidance`).
- Async-task orchestration (`start_async_task`, `check_async_task`).
- Any MCP tools passed in via `mcp_tools=`.

**Seven built-in slash commands** — dispatched by [`CommandDispatcher`](src/langgraph_kit/core/commands/dispatcher.py):

| Command | Effect |
|---|---|
| `/help` | Lists available commands |
| `/memory` | Inspects persistent memory for the current scope |
| `/context` | Shows current context-pressure state and token estimate |
| `/compact` | Forces a microcompaction pass |
| `/status` | Reports agent / thread / pressure status |
| `/tools` | Lists registered tools with risk levels |
| `/skills` | Lists discovered skills |

**Plus** — composite backend factory (memories + notes + state), [Langfuse observability](src/langgraph_kit/observability.py) hooks, automatic persistence across checkpointer+store, and conditional section activation for `memory`, `orchestration`, `deferred_tools`, `skills`, and `async_tasks`.

See [docs/agents/reference-deep-agent.md](docs/agents/reference-deep-agent.md) for the full breakdown.

## Installation

```bash
# Core package (from PyPI once published — currently pre-release from GitHub)
uv add "langgraph-kit @ git+https://github.com/allada-homelab/langgraph-kit@v0.9.0"

# The reference deep agent needs the `deepagents` extra plus one LLM provider
uv add "langgraph-kit[deepagents,anthropic] @ git+https://github.com/allada-homelab/langgraph-kit@v0.9.0"

# The full kitchen sink
uv add "langgraph-kit[all] @ git+https://github.com/allada-homelab/langgraph-kit@v0.9.0"
```

### Optional extras

| Extra | Installs | Use when... |
|---|---|---|
| `openai` | `langchain-openai` | using GPT models (default — also covers OpenAI-compatible endpoints) |
| `anthropic` | `langchain-anthropic` | using `claude-*` models |
| `google` | `langchain-google-genai` | using `gemini-*` models |
| `postgres` | `langgraph-checkpoint-postgres` | running against PostgreSQL in production |
| `deepagents` | `deepagents` | using `reference-deep-agent` or `coding-agent` |
| `mcp` | `langchain-mcp-adapters` | integrating MCP servers as tools |
| `mcp-server` | `mcp` | exposing your agent *as* an MCP server |
| `fastapi` | `fastapi` | using the built-in REST router |
| `agui` | `ag-ui-protocol` | streaming via the AG-UI protocol |
| `a2a` | `a2a-sdk` | agent-to-agent protocol support |
| `langfuse` | `langfuse` | enabling Langfuse tracing |
| `all` | everything above | local development, demos |

## Quickstart

### 1. Configure at startup

```python
from langgraph_kit import AgentConfig, configure

configure(AgentConfig(
    llm_model="claude-sonnet-4-6",          # provider auto-detected from prefix
    llm_api_key="sk-ant-...",
    database_url="sqlite:///checkpoints.db",  # or postgresql://...
))
```

### 2. Register the built-in agents

```python
from langgraph_kit import create_persistence
from langgraph_kit.graphs import register_all

async with create_persistence() as (checkpointer, store):
    register_all(checkpointer, store, mcp_tools=[])
    # echo-agent, basic-deep-agent, reference-deep-agent, coding-agent,
    # and supervisor-agent are all registered.
```

### 3. Stream a conversation

```python
import uuid
from langgraph_kit import get, stream_agent_events

graph = get("reference-deep-agent")
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
input_data = {"messages": [{"role": "user", "content": "Hello!"}]}

async for event in stream_agent_events(graph, input_data, config):
    print(event, end="")
```

### 4. Or expose everything via FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from langgraph_kit import AgentConfig, configure, create_persistence
from langgraph_kit.contrib.fastapi import create_agent_router
from langgraph_kit.graphs import register_all

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure(AgentConfig(llm_model="claude-sonnet-4-6", llm_api_key="sk-ant-..."))
    async with create_persistence() as (checkpointer, store):
        register_all(checkpointer, store, mcp_tools=[])
        app.state.store = store
        yield

app = FastAPI(lifespan=lifespan)
app.include_router(
    create_agent_router(get_current_user=your_auth_dependency),
    prefix="/api/v1",
)
```

That's it. You now have `GET /api/v1/agents/`, `POST /api/v1/agents/{id}/stream` (SSE), `POST /api/v1/agents/{id}/invoke`, thread-state endpoints, a message queue, HITL resume, and checkpoint branching — full list in [docs/integrations/fastapi.md](docs/integrations/fastapi.md).

## Usage guides

Each subsystem is independent and can be used on its own. The examples below are intentionally minimal — follow the links for full docs.

### Memory

Typed, scoped, persistent knowledge that survives across conversations. Five agent-callable CRUD tools, LLM-powered auto-extraction after each turn, and background consolidation to merge near-duplicates and prune stale records.

Runnable demo: [`examples/memory_save_recall.py`](examples/memory_save_recall.py) — creates three typed records, lists them by scope, updates one, and runs a keyword search. Single source of truth: this file is also what the docs site renders for the memory page.

```bash
uv run python -m examples.memory_save_recall
```

See [docs/memory/overview.md](docs/memory/overview.md), [extraction](docs/memory/extraction.md), [consolidation](docs/memory/consolidation.md), [shared-memory](docs/memory/shared-memory.md), [session-notebook](docs/memory/session-notebook.md).

### Tools with capability metadata

Tools aren't just callables — they carry risk levels, profile/worker filters, tags, and prompt guidance. The registry supports filtering and compilation by max risk so a worker only sees what it's allowed to call.

Runnable demo: [`examples/tools_register_capability.py`](examples/tools_register_capability.py) — registers three tools across `READ_ONLY`, `MUTATING`, and `DESTRUCTIVE` risk levels, then filters the compiled list.

```bash
uv run python -m examples.tools_register_capability
```

See [docs/tools/overview.md](docs/tools/overview.md), [capability](docs/tools/capability.md), [registry](docs/tools/registry.md), [memory-tools](docs/tools/memory-tools.md), [worktree-tools](docs/tools/worktree-tools.md).

### Prompt assembly

Layered sections + context providers, ordered stable-first for prompt-cache efficiency.

Runnable demo: [`examples/prompt_assembly_sections.py`](examples/prompt_assembly_sections.py) — registers stable / volatile / conditional sections at different priorities, shows what gets included under different `conditions` sets.

```bash
uv run python -m examples.prompt_assembly_sections
```

See [docs/prompt-assembly/overview.md](docs/prompt-assembly/overview.md).

### Context pressure & compaction

The `PressureMonitor` runs every turn; the `PressureMiddleware` automatically applies the chosen mitigation. Thresholds: 70% → microcompact large tool outputs, 85% → LLM-driven full compaction, 3 failures → circuit-break.

See [docs/context-management/overview.md](docs/context-management/overview.md).

### Slash commands

```python
from langgraph_kit.core.commands.dispatcher import CommandDispatcher, CommandResult

dispatcher = CommandDispatcher()

async def my_handler(args: str, context) -> CommandResult:
    return CommandResult(output=f"You said: {args}", handled=True)

dispatcher.register("/echo", my_handler)
```

The `CommandMiddleware` intercepts any `/`-prefixed user message and short-circuits the LLM on handled commands. Built-ins are registered automatically by [`build_command_dispatcher`](src/langgraph_kit/core/graph_builder/commands.py).

See [docs/commands/overview.md](docs/commands/overview.md).

### Multi-agent orchestration

Declarative worker definitions for the deepagents `task` tool, plus `start_async_task` / `check_async_task` for fire-and-forget background work and a store-backed per-thread message queue for busy threads.

Runnable demo: [`examples/orchestration_workers.py`](examples/orchestration_workers.py) — inspects the bundled `GENERAL_WORKERS` and `CODING_WORKERS` lists and shows how to extend them with a domain-specific worker.

```bash
uv run python -m examples.orchestration_workers
```

See [docs/orchestration/overview.md](docs/orchestration/overview.md).

### Human-in-the-loop

Pause-and-resume approval via LangGraph's `interrupt()` primitive, surfaced through the kit's `approve_action` tool and `/resume` HTTP endpoint.

Runnable demo: [`examples/hitl_approval_flow.py`](examples/hitl_approval_flow.py) — wires a tiny custom graph that interrupts on a `delete_branch` request, then resumes with `Command(resume={"type": "accept"})`.

```bash
uv run python -m examples.hitl_approval_flow
```

See [docs/hitl/overview.md](docs/hitl/overview.md).

### UI events & artifacts

Rich, typed events alongside token stream. Emitted via sentinel-prefixed tool outputs that the streaming layer converts to typed SSE events.

| Tool | SSE key | Use for |
|---|---|---|
| `create_artifact` | `artifact` | Code blocks, markdown, diagrams, tables |
| `emit_progress` | `progress` | Step-by-step progress on multi-step tasks |
| `suggest_actions` | `suggestions` | 2–4 clickable follow-up buttons |
| `add_citation` | `citation` | Collapsible source cards for files, docs, URLs |

See [docs/ui-events/overview.md](docs/ui-events/overview.md).

### Skills (progressive disclosure)

Drop `SKILL.md` files into a skills directory. Agents discover and load them on demand via `discover_skills` and `get_skill_guidance`, keeping the base system prompt small.

See [docs/skills/overview.md](docs/skills/overview.md).

### MCP & plugins

- **MCP servers** — configure via `AgentConfig.mcp_servers` (JSON string) or pass `mcp_tools=[...]` into the builder. Each tool is wrapped as a native `ToolCapability`.
- **Python plugins** — drop `.py` files with a `contribute(registry)` function into `AgentConfig.plugins_dir`.

See [docs/plugins/overview.md](docs/plugins/overview.md).

### Scaffolding a new agent

```bash
uv run python -m langgraph_kit.cli new my-agent --output-dir ./agents/
```

Generates a complete template with prompt sections, worker definitions, tool registration, middleware stack, backend factory, and a `build_graph()` that follows the standard contract.

See [docs/cli/reference.md](docs/cli/reference.md).

### Evaluation

Rule-based and model-graded evaluation with a runner and report module — lives under [`src/langgraph_kit/evals/`](src/langgraph_kit/evals/).

See [docs/evals/overview.md](docs/evals/overview.md).

## Architecture

```
src/langgraph_kit/
├── _config.py        AgentConfig + configure()
├── llm.py            Multi-provider LLM factory
├── persistence.py    Checkpointer + Store factory
├── registry.py       Agent ID → graph mapping
├── streaming.py      SSE event streaming
├── observability.py  Langfuse integration
├── cli.py            Agent scaffolding
│
├── core/             Composable building blocks
│   ├── memory/       Persistent memory, consolidation, shared
│   ├── tools/        Capability model + registry + worktree tools
│   ├── commands/     Slash-command dispatcher
│   ├── context_management/  Pressure monitor, compaction
│   ├── prompt_assembly/     Section-based composer
│   ├── orchestration/       Workers, async tasks, queue
│   ├── resilience/          Completion guard, empty turn, tool error
│   ├── hitl/                Interrupt-based approval
│   ├── skills/              SKILL.md discovery
│   ├── plugins/             MCP + plugin loader
│   └── graph_builder/       Assembly factories
│
├── graphs/           Agent implementations
│   ├── echo_agent.py
│   ├── basic_deep_agent.py
│   ├── reference_deep_agent.py  ← clone this
│   ├── coding_agent.py          ← canonical extension example
│   └── supervisor_agent.py
│
├── contrib/          Optional integrations (fastapi, agui, a2a, mcp_server)
└── evals/            Evaluation framework
```

Full walkthrough: [docs/architecture/overview.md](docs/architecture/overview.md).

### Extension points

| Extension | Mechanism |
|---|---|
| New agent | Implement `build_graph(checkpointer, store)` and `register(...)` it |
| New tool | `registry.register(ToolCapability(...))` |
| New command | `dispatcher.register("/foo", handler)` |
| New prompt section | `sections.register(PromptSection(...))` |
| New context provider | Implement the `ContextProvider` protocol |
| New middleware | Subclass `_AgentMiddleware` |
| New skill | Add a `SKILL.md` file |
| MCP tools | Configure `AgentConfig.mcp_servers` |
| Python plugins | Drop a `.py` with `contribute()` into `plugins_dir` |

## Documentation

Full docs are rendered from [`docs/`](docs/) via MkDocs — start at [docs/index.md](docs/index.md). Highlights:

- [Architecture Overview](docs/architecture/overview.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Configuration](docs/getting-started/configuration.md)
- [Reference Deep Agent](docs/agents/reference-deep-agent.md)
- [Public API](docs/api-reference/public-api.md)
- [SSE Event Types](docs/api-reference/sse-events.md)
- [Store Namespaces](docs/api-reference/store-namespaces.md)

## Development

```bash
git clone https://github.com/allada-homelab/langgraph-kit
cd langgraph-kit
uv sync --extra dev

# Standard loop
just test        # pytest
just lint        # ruff check + codespell
just fmt         # ruff format
just typecheck   # basedpyright
just pre-commit  # all of the above
just build       # hatchling sdist + wheel
```

### Integration testing with a generated app

The test app is generated from [python-template](https://github.com/allada-homelab/python-template) via Copier:

```bash
uv tool install copier
bash scripts/setup-testapp.sh
cd testapp && uv run pytest backend/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

[AGPL-3.0-or-later](LICENSE). If that's a problem for commercial use, open an issue to discuss licensing.
