---
name: pre-push
description: Run the full pre-push gate (pre-commit, basedpyright, pytest) and surface failures. Use when finishing work before `git push`.
disable-model-invocation: true
---

# pre-push

Run the gate that CI enforces, in order, and stop on the first failure. CI re-runs all of these; catching them here saves a red-CI cycle.

## Steps

Run sequentially — do not parallelize. Each step's output informs the next.

1. `just pre-commit` — runs all pre-commit hooks (codespell, ruff check + autofix, ruff format, EOF/whitespace fixers).
2. `just typecheck` — runs `basedpyright` in standard mode against `src/` and `tests/`.
3. `just test` — runs pytest with `filterwarnings=["error"]` and the coverage source registered.

Stop on the first failure and report the failing step with the relevant excerpt of output.

## Common failures

- **pre-commit codespell**: real word? Add it to `[tool.codespell]` (or inline ignore). Typo? Fix it.
- **pre-commit ruff**: most issues auto-fix on first run; re-stage and re-run.
- **basedpyright `reportMissingImports`**: usually means an extra is needed. Check `[project.optional-dependencies]` in `pyproject.toml` and run `uv sync --all-extras --all-groups`.
- **pytest warning-as-error**: the project sets `filterwarnings=["error"]`. Prefer upgrading the offending dependency. Only add a narrow `filterwarnings` exemption (specific message + module) with a comment explaining why no fix exists. (See the `warning-investigator` agent for the full triage workflow.)
- **pytest coverage failure**: only fires when `coverage run` is used (not on plain `just test`). If hit, see the `coverage-floor-guardian` agent — never lower `fail_under`.

## Reporting

Be terse: pass/fail per step, the first failing line with its file:line, and a one-line suggested fix. The user is about to push; they need the next action, not a postmortem.
