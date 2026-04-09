"""Agent invocation config and observability helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from langgraph_kit._config import get_config

logger = logging.getLogger(__name__)


@runtime_checkable
class UserInfo(Protocol):
    """Minimal user interface for agent run metadata."""

    id: Any
    email: str


# ---------------------------------------------------------------------------
# Langfuse helpers
# ---------------------------------------------------------------------------


def langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is fully configured."""
    config = get_config()
    return bool(
        config.langfuse_tracing_enabled
        and config.langfuse_host
        and config.langfuse_public_key
        and config.langfuse_secret_key
    )


def init_langfuse() -> Any | None:
    """Initialize the shared Langfuse client if tracing is configured."""
    if not langfuse_enabled():
        logger.info("Langfuse disabled: missing host or API keys")
        return None

    from langfuse import Langfuse  # pyright: ignore[reportMissingModuleSource]

    config = get_config()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
        environment=config.langfuse_tracing_environment or config.environment,
        release=config.langfuse_release or None,
        tracing_enabled=config.langfuse_tracing_enabled,
    )
    logger.info(
        "Langfuse initialized",
        extra={
            "langfuse_host": config.langfuse_host,
            "langfuse_environment": config.langfuse_tracing_environment
            or config.environment,
        },
    )
    return client


def create_langfuse_handler() -> Any | None:
    """Create a per-request LangChain callback handler."""
    if not langfuse_enabled():
        return None

    from langfuse.langchain import (  # pyright: ignore[reportMissingModuleSource]
        CallbackHandler,
    )

    config = get_config()
    logger.info("Creating Langfuse callback handler for agent run")
    return CallbackHandler(public_key=config.langfuse_public_key)


def flush_langfuse() -> None:
    """Flush any pending Langfuse spans without failing the request."""
    if not langfuse_enabled():
        return

    try:
        from langfuse import Langfuse  # pyright: ignore[reportMissingModuleSource]

        Langfuse().flush()
    except Exception:
        logger.debug("Langfuse flush failed", exc_info=True)


def shutdown_langfuse() -> None:
    """Flush and close the Langfuse client on app shutdown."""
    if not langfuse_enabled():
        return

    try:
        from langfuse import Langfuse  # pyright: ignore[reportMissingModuleSource]

        Langfuse().shutdown()
    except Exception:
        logger.debug("Langfuse shutdown failed", exc_info=True)


# ---------------------------------------------------------------------------
# Agent run config builder
# ---------------------------------------------------------------------------


def build_agent_run_config(
    *,
    agent_id: str,
    thread_id: str,
    current_user: UserInfo,
    endpoint: str,
) -> dict[str, Any]:
    """Build a LangChain/LangGraph runnable config for an agent request."""
    cfg = get_config()
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"{agent_id}.{endpoint}",
        "tags": ["agents", agent_id, endpoint],
        "metadata": {
            "agent_id": agent_id,
            "thread_id": thread_id,
            "endpoint": endpoint,
            "environment": cfg.environment,
            "user_id": str(current_user.id),
            "user_email": current_user.email,
        },
    }

    handler = create_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
        logger.info("Langfuse callback attached", extra={"agent_id": agent_id})
    else:
        logger.info("Langfuse callback not attached", extra={"agent_id": agent_id})

    return config
