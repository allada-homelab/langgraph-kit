"""Package-level configuration for langgraph-kit."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


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


def configure_from_settings(
    settings: Any,
    *,
    field_map: dict[str, str] | None = None,
) -> AgentConfig:
    """Build and apply AgentConfig by matching fields from a settings object.

    Performs case-insensitive matching of AgentConfig field names against
    attributes on *settings*. For example, ``AgentConfig.llm_model`` matches
    ``settings.LLM_MODEL`` or ``settings.llm_model``.

    Parameters
    ----------
    settings:
        Any object with attributes (typically a pydantic-settings instance).
    field_map:
        Optional explicit overrides, e.g.
        ``{"database_url": "SQLALCHEMY_DATABASE_URI"}``.
        Keys are AgentConfig field names, values are attribute names on
        *settings*. These take priority over auto-matching.

    Returns
    -------
    AgentConfig
        The config that was applied (also accessible via ``get_config()``).
    """
    overrides = field_map or {}

    # Build case-insensitive lookup of settings attributes
    settings_attrs: dict[str, str] = {}
    for attr in dir(settings):
        if attr.startswith("_"):
            continue
        try:
            if not callable(getattr(type(settings), attr, None)):
                settings_attrs[attr.lower()] = attr
        except Exception:  # noqa: BLE001
            continue

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(AgentConfig):
        name = field.name

        # 1. Explicit field_map override
        if name in overrides:
            mapped = overrides[name]
            if hasattr(settings, mapped):
                kwargs[name] = _coerce(field, getattr(settings, mapped))
                continue

        # 2. Exact match
        if hasattr(settings, name):
            kwargs[name] = _coerce(field, getattr(settings, name))
            continue

        # 3. UPPER_CASE match (common for env-var style settings)
        upper = name.upper()
        if hasattr(settings, upper):
            kwargs[name] = _coerce(field, getattr(settings, upper))
            continue

        # 4. Case-insensitive scan
        canon = name.lower()
        if canon in settings_attrs:
            kwargs[name] = _coerce(field, getattr(settings, settings_attrs[canon]))
            continue

        # No match — use AgentConfig default

    config = AgentConfig(**kwargs)
    configure(config)
    return config


def _coerce(field: dataclasses.Field[Any], value: Any) -> Any:
    """Coerce *value* to match *field*'s type annotation.

    Handles pydantic URL types and other objects that need ``str()``
    conversion for AgentConfig's str-typed fields.
    """
    if field.type == "str" and not isinstance(value, str):
        return str(value)
    return value
