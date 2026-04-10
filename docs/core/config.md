# Configuration Module

**Source:** `src/langgraph_kit/_config.py`

The configuration module provides a frozen dataclass and two functions for package-level settings management.

## API

### AgentConfig

```python
@dataclass(frozen=True)
class AgentConfig:
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_api_key: str = ""
    database_url: str = "sqlite:///checkpoints.db"
    environment: str = "local"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_tracing_enabled: bool = False
    langfuse_tracing_environment: str = ""
    langfuse_release: str = ""
    mcp_servers: str = ""
    plugins_dir: str = ""
```

The dataclass is **frozen** — instances are immutable after creation. This prevents accidental runtime mutation of configuration values.

### configure(config)

```python
def configure(config: AgentConfig) -> None
```

Sets the package-level config singleton. Must be called **once** at application startup, before any agent code runs.

### get_config()

```python
def get_config() -> AgentConfig
```

Returns the current package-level config. Used internally by all modules that need configuration values. If `configure()` has not been called, returns the default `AgentConfig()`.

## Usage Pattern

```python
# At startup (e.g., FastAPI lifespan)
from langgraph_kit import AgentConfig, configure
configure(AgentConfig(llm_model="claude-sonnet-4-20250514", llm_api_key="sk-ant-..."))

# In any module
from langgraph_kit._config import get_config
config = get_config()
model = config.llm_model  # "claude-sonnet-4-20250514"
```

## Thread Safety

The config is a module-level global. It should be set once before concurrent access begins. The frozen dataclass ensures no mutations occur after initialization.
