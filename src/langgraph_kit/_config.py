"""Package-level configuration for langgraph-kit."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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

    # Token budget (0 = unlimited)
    token_budget_per_thread: int = 0

    # Execution trace export
    trace_export_enabled: bool = False

    # Graceful shutdown: max seconds to wait for in-flight async sub-agent
    # tasks before cancelling them during FastAPI lifespan teardown.
    # Set to 0 to skip draining (cancel immediately).
    shutdown_timeout_seconds: float = 30.0

    # Memory: optional async embedding function for semantic search.
    # When None (default), `PersistentMemoryManager.search` falls back to
    # keyword token-overlap scoring. When provided, records are indexed on
    # create/update and search ranks by cosine similarity. The callable is
    # invoked batch-style: takes a list of texts, returns a list of vectors.
    # No silent semantic/keyword mixing — the presence of the callable is
    # the only signal that semantic search is enabled.
    memory_embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = (
        dataclasses.field(default=None, compare=False)
    )

    # Security: inbound prompt-injection scanner mode.
    # ``"warn"`` (default) scans every user message and logs / flags
    # detections without changing agent behaviour. ``"off"`` disables
    # the scan entirely. The richer ``"quarantine"`` mode (narrow tool
    # surface for the affected turn) is tracked separately and will
    # layer on top of this default in a follow-up.
    prompt_injection_mode: str = "warn"

    # Security: outbound assistant-message scanner mode.
    # ``"redact"`` (default) replaces matched PII / secrets with
    # ``[REDACTED]`` before the message is shown to the user.
    # ``"warn"`` flags without mutating (useful for shadow-mode
    # rollouts to gather detection metrics first). ``"off"`` disables.
    output_safety_mode: str = "redact"

    def __repr__(self) -> str:
        """Mask secrets in repr to prevent accidental leakage in logs.

        Only fields that are genuinely secret are masked. Langfuse public
        keys are intentionally public (they identify the project to the
        Langfuse API — they're not credentials), so leaving them readable
        helps log triage. Short secrets (<=8 chars) are fully masked so
        a 1-char secret ``"x"`` doesn't render as ``"x***"``.
        """
        fields = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if f.name in ("llm_api_key", "langfuse_secret_key") and val:
                val = "****" if len(val) <= 8 else val[:4] + "..."
            fields.append(f"{f.name}={val!r}")
        return f"AgentConfig({', '.join(fields)})"


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
        except Exception:  # noqa: S112 — intentional: dir() can yield attrs that raise on getattr
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


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Result of running :func:`validate_config` against an :class:`AgentConfig`.

    Two-tier surface: ``errors`` would prevent the kit from running
    correctly (bad URL scheme, negative budget); ``warnings`` are
    suspicious but not fatal (Langfuse public key set without secret,
    embedding-fn unset when memory is heavily used).

    Frozen so callers can hand the report around without worrying
    about downstream mutation. Helpers like :py:meth:`is_ok` keep the
    common "did anything fail?" check terse.
    """

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_ok(self) -> bool:
        """``True`` iff no errors. Warnings don't fail this gate."""
        return not self.errors


_VALID_DB_SCHEMES = ("sqlite://", "postgresql://", "postgresql+psycopg://")
"""URL schemes ``create_persistence`` recognizes. SQLite for dev,
Postgres (with the optional psycopg driver suffix) for production.
Unknown schemes silently fall through to "checkpointer + store can't
be built" failures deep in the stack — better to catch up front."""


_VALID_PROMPT_INJECTION_MODES = ("off", "warn")
"""Modes accepted by :pyattr:`AgentConfig.prompt_injection_mode`.

The ``"quarantine"`` mode is reserved for a follow-up PR; if a caller
sets it today the scanner falls back to ``"warn"`` semantics. Validate
against the actual accepted set so typos are caught."""


_VALID_OUTPUT_SAFETY_MODES = ("off", "warn", "redact")
"""Modes accepted by :pyattr:`AgentConfig.output_safety_mode`."""


def validate_config(cfg: AgentConfig) -> ValidationReport:
    """Surface configuration mistakes without raising.

    Returns a :class:`ValidationReport` with separate ``errors`` and
    ``warnings`` tuples. Caller decides what to do — the kit's CLI
    ``validate-config`` subcommand prints both and exits non-zero on
    errors; production callers can wire this into their startup path
    for fail-loud-and-early behavior.

    Pure function — no I/O, no side effects. A separate
    ``--check-connections`` pass that actually opens the database is
    deferred to a follow-up; it would change this contract.

    Checks performed:

    1. ``database_url`` uses a recognized scheme.
    2. ``llm_model`` is non-empty (every code path needs one).
    3. ``token_budget_per_thread`` is non-negative.
    4. ``shutdown_timeout_seconds`` is non-negative.
    5. ``prompt_injection_mode`` is one of :data:`_VALID_PROMPT_INJECTION_MODES`.
    6. ``output_safety_mode`` is one of :data:`_VALID_OUTPUT_SAFETY_MODES`.
    7. Langfuse keys appear in matching pairs; loud warning when only
       one half is set (the kit silently disables tracing in that
       case, which is a reliable source of confusion).
    8. ``langfuse_tracing_enabled=True`` requires both keys; warns
       when tracing is on but credentials are incomplete.
    9. ``mcp_servers`` parses as JSON when non-empty.
    """
    import json  # local import — keeps validation cheap when not invoked

    errors: list[str] = []
    warnings: list[str] = []

    # 1. database_url scheme
    if not cfg.database_url:
        errors.append("database_url is empty; set a sqlite:// or postgresql:// URL")
    elif not cfg.database_url.startswith(_VALID_DB_SCHEMES):
        errors.append(
            f"database_url uses an unsupported scheme: {cfg.database_url!r} "
            f"(expected one of {_VALID_DB_SCHEMES})"
        )

    # 2. llm_model
    if not cfg.llm_model:
        errors.append("llm_model is empty; the kit requires a default model")

    # 3-4. Numeric bounds
    if cfg.token_budget_per_thread < 0:
        errors.append(
            f"token_budget_per_thread must be >= 0 (0 = unlimited); "
            f"got {cfg.token_budget_per_thread}"
        )
    if cfg.shutdown_timeout_seconds < 0:
        errors.append(
            f"shutdown_timeout_seconds must be >= 0 "
            f"(0 = cancel immediately); got {cfg.shutdown_timeout_seconds}"
        )

    # 5-6. Mode strings
    if cfg.prompt_injection_mode not in _VALID_PROMPT_INJECTION_MODES:
        errors.append(
            f"prompt_injection_mode={cfg.prompt_injection_mode!r} "
            f"is not one of {_VALID_PROMPT_INJECTION_MODES}"
        )
    if cfg.output_safety_mode not in _VALID_OUTPUT_SAFETY_MODES:
        errors.append(
            f"output_safety_mode={cfg.output_safety_mode!r} "
            f"is not one of {_VALID_OUTPUT_SAFETY_MODES}"
        )

    # 7. Langfuse pair consistency
    has_public = bool(cfg.langfuse_public_key)
    has_secret = bool(cfg.langfuse_secret_key)
    if has_public and not has_secret:
        warnings.append(
            "langfuse_public_key is set but langfuse_secret_key is empty; "
            "Langfuse tracing will be disabled at runtime"
        )
    elif has_secret and not has_public:
        warnings.append(
            "langfuse_secret_key is set but langfuse_public_key is empty; "
            "Langfuse tracing will be disabled at runtime"
        )

    # 8. Tracing-enabled guard
    if cfg.langfuse_tracing_enabled:
        if not has_public or not has_secret:
            errors.append(
                "langfuse_tracing_enabled=True requires both "
                "langfuse_public_key and langfuse_secret_key"
            )
        if not cfg.langfuse_host:
            errors.append("langfuse_tracing_enabled=True requires a langfuse_host URL")

    # 9. mcp_servers JSON
    if cfg.mcp_servers:
        try:
            json.loads(cfg.mcp_servers)
        except ValueError as exc:
            errors.append(f"mcp_servers is not valid JSON: {exc}")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))
