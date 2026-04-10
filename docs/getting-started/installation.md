# Installation

langgraph-kit is distributed as a Python package built with [Hatchling](https://hatch.pypa.io/). It requires **Python 3.13**.

## From GitHub

```bash
# Core package
uv add "langgraph-kit @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"

# With FastAPI integration
uv add "langgraph-kit[fastapi] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"

# With Anthropic Claude support
uv add "langgraph-kit[anthropic] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"

# With Google Gemini support
uv add "langgraph-kit[google] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"

# Multiple extras
uv add "langgraph-kit[fastapi,anthropic] @ git+https://github.com/allada-homelab/langgraph-kit@v0.1.0"
```

## From PyPI

```bash
uv add langgraph-kit
uv add "langgraph-kit[fastapi]"
```

## Optional Extras

| Extra | Installs | Purpose |
|-------|----------|---------|
| `fastapi` | `fastapi>=0.100` | FastAPI router factory for HTTP endpoints |
| `anthropic` | `langchain-anthropic>=0.3` | Claude model support (`claude-*` models) |
| `google` | `langchain-google-genai>=2.0` | Gemini model support (`gemini-*` models) |
| `dev` | pytest, pytest-asyncio, coverage, ruff | Development and testing tools |

## Core Dependencies

These are installed automatically with the base package:

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | `>=2.0` | Data validation and serialization |
| `langgraph` | `>=0.3` | Graph execution framework |
| `langgraph-checkpoint-postgres` | `>=2.0` | PostgreSQL persistence |
| `langgraph-checkpoint-sqlite` | `>=2.0` | SQLite persistence |
| `langchain-core` | `>=0.3` | LLM abstractions |
| `langchain-openai` | `>=0.3` | OpenAI-compatible models (default) |
| `deepagents` | `>=0.4` | Multi-agent framework |
| `langfuse` | `>=4.0` | Observability and tracing |
| `langchain-mcp-adapters` | `>=0.2` | Model Context Protocol integration |
| `structlog` | `>=24.0` | Structured logging |

## Development Setup

```bash
git clone https://github.com/allada-homelab/langgraph-kit
cd langgraph-kit

# Install all dependencies including dev extras
uv sync --extra dev

# Verify installation
uv run pytest
uv run ruff check src/
```

## Verifying Installation

```python
from langgraph_kit import AgentConfig, configure
configure(AgentConfig(llm_model="gpt-4o-mini"))

from langgraph_kit import build_llm
llm = build_llm()
print(type(llm))  # <class 'langchain_openai.chat_models.base.ChatOpenAI'>
```
