# Contributing to langgraph-kit

Thanks for your interest. This toolkit wraps [LangGraph](https://github.com/langchain-ai/langgraph)
with reusable building blocks for agent memory, tools, orchestration,
context management, and evaluation.

## Local setup

```bash
git clone https://github.com/allada-homelab/langgraph-kit
cd langgraph-kit
uv sync --all-extras --all-groups
just test
```

Python 3.11–3.14 are supported; CI runs on all four.

## Common tasks

`just --list` shows everything. The most-used recipes:

| Command | What it does |
| --- | --- |
| `just test` | Run pytest |
| `just coverage` | Run coverage and print a missing-lines report |
| `just lint` / `just lint-fix` | Ruff lint, optionally autofix |
| `just fmt` / `just fmt-check` | Ruff format |
| `just typecheck` | basedpyright |
| `just pre-commit` | Run the full pre-commit suite |
| `just docs` | Build the mkdocs site locally |
| `just docs-serve` | Serve the mkdocs site at http://localhost:8000 |
| `just build` | Build wheel + sdist with `uv build` |
| `just testapp` | Regenerate the copier-based test app for integration work |

## Test layout

```
tests/
  conftest.py              # MockStore + shared async fixtures
  test_*.py                # One module per subsystem
```

Tests are async-first (`asyncio_mode = "auto"`). The `filterwarnings = ["error"]`
rule promotes warnings to hard failures — if a dependency emits a deprecation
warning, either upgrade or add a narrow, documented exemption in
`pyproject.toml`.

Coverage has a floor enforced by CI (see `[tool.coverage.report] fail_under`
in `pyproject.toml`). Raise it when you add tests; never lower it to paper
over a regression.

## Code style

- Module layout: `src/langgraph_kit/<subsystem>/`. Public API re-exports from
  `src/langgraph_kit/__init__.py`.
- `from __future__ import annotations` at the top of every module;
  `keep-runtime-typing = true` is set so runtime tools can still read the
  hints.
- Lint: `ruff check`. Format: `ruff format`. Type-check: `basedpyright`. All
  three must be clean before merge.
- Test one subsystem per module. Class-based grouping (`TestX`) is the
  convention already used across `tests/`.

## Releasing

Releases are tag-driven via PyPI Trusted Publishing (no manual API tokens).

The preferred path is the GitHub UI:

1. Update `CHANGELOG.md` — move items out of `## [Unreleased]` into a new
   `## [X.Y.Z] — YYYY-MM-DD` section.
2. Draft a new release in GitHub, pick the tag (`vX.Y.Z`), and click publish.

The `release` workflow will:

- Derive the version from the tag and overwrite `src/langgraph_kit/__version__.py`
  so the wheel metadata always matches the release label.
- Build with `uv build`, publish to PyPI via OIDC trusted publishing, attach
  artifacts to the GitHub release, and commit the version bump back to `main`.

If the CHANGELOG has a section for the version, it's used as the GitHub
release body; otherwise GitHub's auto-generated notes are used.

The fallback `git tag vX.Y.Z && git push origin vX.Y.Z` path also works but
skips the post-release bump on `main`.

## Questions / bugs

Open an issue: <https://github.com/allada-homelab/langgraph-kit/issues>.
