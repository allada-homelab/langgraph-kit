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

# Release
build:
    uv build

release-test:
    uv publish --index testpypi
