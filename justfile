default:
    @just --list

# Python
install:
    uv sync --all-extras --all-groups

test:
    uv run pytest

coverage:
    uv run coverage run -m pytest && uv run coverage report --show-missing

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check --fix .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

typecheck:
    uv run basedpyright

pre-commit:
    uv run pre-commit run --all-files

# Docs
docs:
    uv run mkdocs build --strict

docs-serve:
    uv run mkdocs serve

# Integration harness (generated test app)
testapp:
    bash scripts/setup-testapp.sh

# Examples — runs every examples/*.py through the hermetic smoke driver.
# Set RUN_NETWORK=1 to also exercise REQUIRES_NETWORK examples.
examples-smoke:
    uv run python -m examples.run_all

# Prompt-bench — internal prompt-optimization harness under tests/prompt_bench/.
# Hermetic by default. Set PROMPT_BENCH_LLM=real and AGENT_LLM_API_KEY for live runs.
prompt-bench-test:
    uv run pytest tests/prompt_bench -v -p no:unraisableexception

prompt-bench-targets:
    uv run python -m tests.prompt_bench.run list-targets

prompt-bench-scenarios target="":
    uv run python -m tests.prompt_bench.run list-scenarios {{ if target != "" { "--target " + target } else { "" } }}

# Phase 0 surfaces signal-check via the CLI; it exits non-zero until the
# Phase 1 PR wires live execution. Remove the leading "-" once that lands.
prompt-bench-signal-check target:
    -uv run python -m tests.prompt_bench.run signal-check --target {{target}}

# Release
build:
    uv build

release-test:
    uv publish --index testpypi
