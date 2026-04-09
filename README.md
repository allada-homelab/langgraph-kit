# langgraph-kit

Reusable LangGraph agent toolkit with memory, tools, and orchestration.

## Installation

```bash
# From GitHub
uv add "langgraph-kit @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"

# With optional extras
uv add "langgraph-kit[fastapi] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"
uv add "langgraph-kit[anthropic] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"
uv add "langgraph-kit[google] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run basedpyright
```

### Integration testing with a generated app

The test app is generated from [python-template](https://github.com/allada-homelab/python-template) via Copier:

```bash
# Requires: uv tool install copier
bash scripts/setup-testapp.sh

# Run integration tests
cd testapp && uv run pytest backend/
```
