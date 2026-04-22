"""Agent invocation config and observability helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from langgraph_kit._config import get_config

logger = logging.getLogger(__name__)

# Module-level Langfuse client — initialized by init_langfuse(), reused by
# flush/shutdown so they operate on the same client that holds the spans.
_langfuse_client: Any = None


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
    global _langfuse_client

    if not langfuse_enabled():
        logger.info("Langfuse disabled: missing host or API keys")
        return None

    from langfuse import Langfuse  # pyright: ignore[reportMissingModuleSource]

    config = get_config()
    _langfuse_client = Langfuse(
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
    return _langfuse_client


def create_langfuse_handler() -> Any | None:
    """Create a per-request LangChain callback handler."""
    if not langfuse_enabled():
        return None

    from langfuse.langchain import (  # pyright: ignore[reportMissingModuleSource]
        CallbackHandler,
    )

    config = get_config()
    logger.info("Creating Langfuse callback handler for agent run")
    # langfuse>=4: CallbackHandler only accepts public_key + trace_context.
    # secret_key/host are read from the globally-configured Langfuse client
    # that init_langfuse() created.
    return CallbackHandler(public_key=config.langfuse_public_key)


def flush_langfuse() -> None:
    """Flush any pending Langfuse spans without failing the request."""
    if _langfuse_client is None:
        return

    try:
        _langfuse_client.flush()
    except Exception:
        logger.warning("Langfuse flush failed", exc_info=True)


def shutdown_langfuse() -> None:
    """Flush and close the Langfuse client on app shutdown."""
    global _langfuse_client

    if _langfuse_client is None:
        return

    try:
        _langfuse_client.shutdown()
    except Exception:
        logger.warning("Langfuse shutdown failed", exc_info=True)
    finally:
        _langfuse_client = None


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

    callbacks: list[Any] = []

    handler = create_langfuse_handler()
    if handler is not None:
        callbacks.append(handler)
        logger.info("Langfuse callback attached", extra={"agent_id": agent_id})

    # Execution trace export
    if cfg.trace_export_enabled:
        from langgraph_kit.core.tracing.handler import TraceCallbackHandler

        trace_handler = TraceCallbackHandler(agent_id=agent_id, thread_id=thread_id)
        callbacks.append(trace_handler)
        config["metadata"]["_trace_handler"] = trace_handler
        logger.info("Trace export attached", extra={"agent_id": agent_id})

    # Token budget tracking
    if cfg.token_budget_per_thread > 0:
        from langgraph_kit.core.cost.callback import TokenTrackingCallback

        budget_callback = TokenTrackingCallback()
        callbacks.append(budget_callback)
        config["metadata"]["_budget_callback"] = budget_callback
        logger.info("Budget tracking attached", extra={"agent_id": agent_id})

    if callbacks:
        config["callbacks"] = callbacks

    return config
