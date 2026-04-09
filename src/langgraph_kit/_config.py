"""Package-level configuration for langgraph-kit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the agent toolkit.

    Consumers call ``configure(AgentConfig(...))`` once at startup.
    Internal modules read values via ``get_config()``.
    """

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_api_key: str = ""

    # Persistence
    database_url: str = "sqlite:///checkpoints.db"

    # Environment
    environment: str = "local"

    # Langfuse observability
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_tracing_enabled: bool = False
    langfuse_tracing_environment: str = ""
    langfuse_release: str = ""

    # MCP servers (JSON string)
    mcp_servers: str = ""

    # Plugin directory (optional — Python files with contribute() functions)
    plugins_dir: str = ""


_config: AgentConfig = AgentConfig()


def configure(config: AgentConfig) -> None:
    """Set the package-level config. Call once at startup."""
    global _config
    _config = config


def get_config() -> AgentConfig:
    """Return the current package-level config."""
    return _config
