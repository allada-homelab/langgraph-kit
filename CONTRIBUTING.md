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

Python 3.11–3.13 are supported; CI runs on all three.

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

## Adding an example

Examples live in [`examples/`](examples/) at the repo root and double as
the docs site's runnable code samples. Each is a standalone Python file
that runs hermetically (no API keys) by default.

Quick start:

1. Pick a feature without a demo — see the table in
   [`examples/README.md`](examples/README.md) and the open Phase 2 / 3
   sub-issues of [#61](https://github.com/allada-homelab/langgraph-kit/issues/61).
2. Create `examples/<name>.py` using the template in `examples/README.md`.
3. Persist any state through `tmp_workspace()` from
   [`examples/_lib.py`](examples/_lib.py) — never `~` or repo root.
4. If the example needs network or a real LLM, declare
   `REQUIRES_NETWORK = True` at module top so the per-PR smoke job
   skips it (the nightly workflow picks it up).
5. Run `just examples-smoke` to confirm it stays green; CI runs the
   same command on every PR.

The hermetic substrate is `langgraph_kit.replay.RecordedChatModel`,
which is also what the e2e suite uses — patterns that work in
`tests/e2e/` work for an example.

## Optimizing a prompt

The kit has ~25 distinct prompts (agent system prompts, prompt-assembly
sections, middleware prompts, worker definitions). They evolve through
the internal **prompt-bench** harness under `tests/prompt_bench/`.

Iteration loop:

1. **Read [tests/prompt_bench/README.md](tests/prompt_bench/README.md)** for the full workflow,
   acceptance bar, and seven variant dimensions to try (one per
   iteration).
2. **Drop a candidate** at `tests/prompt_bench/variants/<target>/<variant_name>.md`.
   Frontmatter (notes, dimension changed) is stripped; everything below
   is the prompt text the overlay injects.
3. **Run the harness.** Hermetic check via `just prompt-bench-test`.
   For real-LLM iteration, populate `.env` with
   `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` (any
   OpenAI-compatible endpoint) and run:
   ```bash
   set -a && source .env && set +a
   uv run python -m tests.prompt_bench.run run \
     --target <target> --variant <variant_name>
   ```
4. **Diff** baseline vs variant; the markdown report goes in the PR.
5. **Strict acceptance bar** — a change ships only when:
   - Pairwise win rate ≥ 60% across decided pairs
   - Two-judge agreement ≥ 70%
   - No rule-based metric regression > 5%
   - Cross-prompt regression suite passes
   - N=5 samples per scenario (re-run high-variance scenarios with N=10)

Do not lower the thresholds (`WIN_RATE_THRESHOLD` /
`JUDGE_AGREEMENT_THRESHOLD` / `METRIC_REGRESSION_TOLERANCE` in
`tests/prompt_bench/diff.py`) per-PR. If a candidate doesn't clear the
bar, it doesn't ship — iterate the prompt instead of relaxing the
gate.

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
  artifacts to the GitHub release, and propagate the version bump back to `main`
  (see "Post-release version bump" below).

If the CHANGELOG has a section for the version, it's used as the GitHub
release body; otherwise GitHub's auto-generated notes are used.

The fallback `git tag vX.Y.Z && git push origin vX.Y.Z` path also works.

### Post-release version bump

The `bump-main` job propagates the new version into `__version__.py` on `main`
so future development starts from the post-release number. Two paths:

- **Direct push** (fast path): used when `secrets.PRE_COMMIT` is set to a PAT
  with bypass permission on `main`'s branch protection. Bumps land immediately
  as a single `[skip ci]` commit. This is the same PAT the pre-commit
  workflow uses to push autofixes back to PR branches.
- **Bump PR** (fallback): when no bypass-capable token is available — or the
  direct push is rejected — the workflow opens a PR titled
  `chore: bump __version__ to X.Y.Z` (labeled `internal`). CI runs against
  the PR; merge it once green. With repo-level "allow auto-merge" enabled this
  can be set up to merge unattended; otherwise it sits open for review.

Either way, published wheels are correct independently of this step (the
build job rewrites `__version__.py` before `uv build`), so a stuck bump PR
never affects PyPI artifacts — only the source-of-truth on `main`.

## Questions / bugs

Open an issue: <https://github.com/allada-homelab/langgraph-kit/issues>.
