# langgraph-kit

A batteries-included toolkit that wraps [LangGraph](https://github.com/langchain-ai/langgraph) with reusable production primitives (memory, tools, prompt assembly, orchestration, HITL, middleware, evals). Alpha — APIs may still evolve before 1.0. AGPL-3.0-or-later.

See [README.md](README.md) for the feature tour, [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow, and [docs/architecture/overview.md](docs/architecture/overview.md) for the deep dive.

## Project tracking

Work is tracked on GitHub Projects: https://github.com/orgs/allada-homelab/projects/1

## Environment & tooling

- **Python:** 3.11 – 3.13 (CI runs all three; target-version in ruff is `py311`).
- **Package manager:** `uv` — use `uv run <cmd>` for anything invoking the project environment. Never invoke `python` / `pytest` / `ruff` directly.
- **Task runner:** `just` (recipes in [justfile](justfile)). `just --list` shows everything.
- **Bootstrap:** `uv sync --all-extras --all-groups` (the `dev` group pulls in `test`, `lint`, `type`, and `langgraph-kit[all]`).

## Before pushing

Always run these locally and fix any findings — catching them here is faster than iterating on red CI:

```bash
just pre-commit   # pre-commit run --all-files
just typecheck    # basedpyright
just test         # pytest
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs: `ruff check`, `ruff format --check`, `basedpyright`, `pytest`, and gated `coverage report` on Python 3.13. Pre-commit also runs codespell + trailing-whitespace/EOF/large-file/merge-conflict checks.

## Code conventions

- **`from __future__ import annotations` at the top of every module.** Ruff `FA` enforces it.
- **`keep-runtime-typing = true`** in ruff — don't rewrite `List[X]` → `list[X]` in runtime-read hints blindly; pyupgrade is scoped to keep runtime type lookups working.
- **Public API re-exports** live in [src/langgraph_kit/__init__.py](src/langgraph_kit/__init__.py). New top-level exports go through `__all__` there.
- **Module layout:** `src/langgraph_kit/<subsystem>/`. Mirror this in `tests/` (one test module per subsystem).
- **Lint selection** (see [pyproject.toml](pyproject.toml)): `F,E,W,I,UP,B,SIM,C4,RUF,S,A,PT,RET,PTH,FA,TC,ARG,T20`. Notable: `S` (bandit), `T20` (no `print`), `PTH` (prefer `pathlib`).
- **Ignores that matter:** `E501` (formatter handles wrapping), `B008` (FastAPI `Depends` default pattern), `TC001` in `src/` (runtime type lookups).
- **basedpyright** runs in `standard` mode (see [pyrightconfig.json](pyrightconfig.json)); must be clean.
- **No emojis** in user-facing output — enforced by [tests/test_no_emojis_in_user_facing_output.py](tests/test_no_emojis_in_user_facing_output.py).

## Testing

- **Async-first:** `asyncio_mode = "auto"` — write `async def test_...` directly; no `@pytest.mark.asyncio` needed.
- **Warnings are errors:** `filterwarnings = ["error"]`. If a dep emits a deprecation warning, upgrade it or add a *narrow, documented* exemption in [pyproject.toml](pyproject.toml) — don't blanket-silence.
- **`xfail_strict = true`:** an xfail that unexpectedly passes is a failure. Remove the marker instead of leaving it.
- **Markers:** `integration` (needs external services), `e2e` (runs a real compiled graph via scripted `RecordedChatModel`). Declare custom markers in `[tool.pytest.ini_options]`.
- **Shared fixtures:** [tests/conftest.py](tests/conftest.py) provides `MockStore` + `mock_store`. E2E has its own [tests/e2e/conftest.py](tests/e2e/conftest.py) (`checkpointer`, `e2e_store`, `patched_build_llm`, scripting helpers like `scripted_llm`, `tool_call_turn`, `answer`).
- **Coverage floor:** `fail_under = 55` in [pyproject.toml](pyproject.toml). Actual is ~90%. Raise the floor as the suite grows; **never lower it** to paper over a regression.
- **Conventions:** one module per subsystem, class-based grouping (`class TestX:`).

## Architecture map

```
src/langgraph_kit/
├── _config.py, llm.py, persistence.py, registry.py, streaming.py, observability.py, cli.py
├── core/             Composable building blocks (memory, tools, commands, context_management,
│                     prompt_assembly, orchestration, resilience, hitl, skills, plugins,
│                     graph_builder, tracing, cost)
├── graphs/           Agent implementations
│   ├── _builder.py              Shared deep-agent skeleton + DEFAULT_RECURSION_LIMIT
│   ├── reference_deep_agent.py  ← clone this as the starting point for new agents
│   ├── coding_agent.py          ← canonical extension pattern
│   ├── basic_deep_agent.py, echo_agent.py, supervisor_agent.py
├── contrib/          Optional integrations (fastapi, agui, a2a, mcp_server)
├── evals/            Evaluation framework (runner, report, rule-based + model-graded metrics)
├── replay/           RecordedChatModel for deterministic replay tests
└── skills/           Bundled SKILL.md files (code-review, research)
```

Extension points (preferred way to add capability):

| Extension         | Mechanism                                                   |
|-------------------|-------------------------------------------------------------|
| New agent         | `build_graph(checkpointer, store)` + `register(...)`        |
| New tool          | `registry.register(ToolCapability(...))`                    |
| New command       | `dispatcher.register("/foo", handler)`                      |
| New prompt section| `sections.register(PromptSection(...))`                     |
| New middleware    | Subclass `_AgentMiddleware`                                 |
| New skill         | Add a `SKILL.md` under a skills directory                   |
| MCP / plugin      | `AgentConfig.mcp_servers` / drop `.py` with `contribute()`  |

## Gotchas

- **Recursion limit = 100, not 25.** `DEFAULT_RECURSION_LIMIT` in [src/langgraph_kit/graphs/_builder.py](src/langgraph_kit/graphs/_builder.py) overrides LangGraph's native default. Deep agents easily burn 25 supersteps per real task (middleware passes, worker round-trips, tool loops). Override per-build (`recursion_limit=500`) or per-run (`config={"recursion_limit": 500}`).
- **`testapp/` is generated, not checked in.** Regenerate via `just testapp` (runs [scripts/setup-testapp.sh](scripts/setup-testapp.sh) which uses Copier against [python-template](https://github.com/allada-homelab/python-template)). Gitignored.
- **Langfuse public keys aren't secrets** — they identify the project to the Langfuse API. `AgentConfig.__repr__` intentionally masks only `llm_api_key` + `langfuse_secret_key`.
- **`AgentConfig` is frozen.** `configure(AgentConfig(...))` is called once at startup; internal modules read via `get_config()`. Don't mutate.
- **Persistence switches on URL scheme.** `sqlite://...` → `AsyncSqliteSaver + InMemoryStore`; `postgresql://...` → `AsyncPostgresSaver + AsyncPostgresStore`. Both wired by `create_persistence()`.
- **`checkpoints.db` is gitignored** (dev-only SQLite file). Don't commit it.
- **Reference lists in [README.md](README.md) are load-bearing.** The middleware order, worker definitions, and slash-command table reflect actual wiring — if you change the builder, update them.

## Docs

MkDocs source in [docs/](docs/), rendered to [site/](site/) via `just docs` (strict mode) or `just docs-serve` (localhost:8000). `site/` is gitignored. Docs workflow: [.github/workflows/docs.yml](.github/workflows/docs.yml).

## Releasing

Tag-driven via PyPI Trusted Publishing — see [CONTRIBUTING.md](CONTRIBUTING.md#releasing). Move items from `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) into a versioned section, draft a GitHub release with tag `vX.Y.Z`, and the `release` workflow handles version bump + publish + back-commit.
