# Configuration

langgraph-kit uses a single frozen dataclass, `AgentConfig`, for all package-level settings. Call `configure()` once at startup; internal modules read values via `get_config()`.

## AgentConfig

```python
from langgraph_kit import AgentConfig, configure

configure(AgentConfig(
    # LLM settings
    llm_model="gpt-4o-mini",       # Model name (provider auto-detected)
    llm_base_url="",               # Custom endpoint (OpenAI-compatible)
    llm_api_key="",                # API key for the LLM provider

    # Persistence
    database_url="sqlite:///checkpoints.db",  # PostgreSQL or SQLite

    # Environment
    environment="local",           # "local", "staging", or "production"

    # Langfuse observability
    langfuse_host="",
    langfuse_public_key="",
    langfuse_secret_key="",
    langfuse_tracing_enabled=False,
    langfuse_tracing_environment="",
    langfuse_release="",

    # MCP servers (JSON string)
    mcp_servers="",

    # Plugin directory
    plugins_dir="",
))
```

## Configuration Fields

### LLM

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_model` | `str` | `"gpt-4o-mini"` | Model identifier. Provider is auto-detected from the name prefix. |
| `llm_base_url` | `str` | `""` | Base URL for OpenAI-compatible endpoints. Leave empty for default provider URLs. |
| `llm_api_key` | `str` | `""` | API key for the selected provider. |

**Provider detection rules:**

| Model prefix | Provider | LangChain class | Extra required |
|-------------|----------|-----------------|----------------|
| `claude-*` | Anthropic | `ChatAnthropic` | `anthropic` |
| `gemini-*` | Google | `ChatGoogleGenerativeAI` | `google` |
| anything else | OpenAI | `ChatOpenAI` | _(none)_ |

### Persistence

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `database_url` | `str` | `"sqlite:///checkpoints.db"` | Connection URL. Prefix determines backend. |

- **`postgresql://...`** — Uses `AsyncPostgresSaver` + `AsyncPostgresStore` (full persistence)
- **`sqlite:///path`** — Uses `AsyncSqliteSaver` + `InMemoryStore` (store data lost on restart)

The `postgresql+psycopg` scheme is automatically normalized to `postgresql` for LangGraph compatibility.

### Langfuse Observability

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `langfuse_host` | `str` | `""` | Langfuse server URL |
| `langfuse_public_key` | `str` | `""` | Langfuse public key |
| `langfuse_secret_key` | `str` | `""` | Langfuse secret key |
| `langfuse_tracing_enabled` | `bool` | `False` | Enable/disable tracing |
| `langfuse_tracing_environment` | `str` | `""` | Environment tag for traces |
| `langfuse_release` | `str` | `""` | Release version tag |

Langfuse is considered enabled when `langfuse_tracing_enabled` is `True` and both `langfuse_public_key` and `langfuse_secret_key` are non-empty.

### Plugins

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_servers` | `str` | `""` | JSON string defining MCP server connections |
| `plugins_dir` | `str` | `""` | Path to directory containing plugin Python files |

## Reading Configuration

```python
from langgraph_kit import get_config

config = get_config()
print(config.llm_model)       # "gpt-4o-mini"
print(config.database_url)    # "sqlite:///checkpoints.db"
```

The config object is frozen (immutable). To change settings, call `configure()` again with a new `AgentConfig` instance.

## Integration with FastAPI

In a FastAPI application, configuration is typically driven by environment variables through pydantic-settings:

```python
from app.core.config import settings
from langgraph_kit import AgentConfig, configure

configure(AgentConfig(
    llm_model=settings.LLM_MODEL,
    llm_base_url=settings.LLM_BASE_URL,
    llm_api_key=settings.LLM_API_KEY,
    database_url=str(settings.SQLALCHEMY_DATABASE_URI),
    langfuse_tracing_enabled=settings.LANGFUSE_TRACING_ENABLED,
    langfuse_public_key=settings.LANGFUSE_PUBLIC_KEY,
    langfuse_secret_key=settings.LANGFUSE_SECRET_KEY,
    langfuse_host=settings.LANGFUSE_HOST,
))
```
