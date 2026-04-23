"""Coverage fill — observability helpers (Langfuse gating + run config).

Everything in ``observability.py`` that's touchable without a live
Langfuse instance: the ``langfuse_enabled`` gate, graceful no-ops
when tracing isn't configured, and the ``build_agent_run_config``
metadata construction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langgraph_kit import observability as obs_mod
from langgraph_kit._config import AgentConfig
from langgraph_kit.observability import (
    build_agent_run_config,
    create_langfuse_handler,
    flush_langfuse,
    init_langfuse,
    langfuse_enabled,
    shutdown_langfuse,
)


def _config(**overrides: Any) -> AgentConfig:
    defaults: dict[str, Any] = {
        "environment": "test",
        "langfuse_tracing_enabled": False,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


class _User:
    id = "user-1"
    email = "u@example.com"


def test_langfuse_disabled_when_keys_missing() -> None:
    with patch(
        "langgraph_kit.observability.get_config",
        return_value=_config(langfuse_tracing_enabled=True, langfuse_host=""),
    ):
        assert langfuse_enabled() is False


def test_langfuse_enabled_when_fully_configured() -> None:
    with patch(
        "langgraph_kit.observability.get_config",
        return_value=_config(
            langfuse_tracing_enabled=True,
            langfuse_host="https://lf.test",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        ),
    ):
        assert langfuse_enabled() is True


def test_init_langfuse_returns_none_when_disabled() -> None:
    with patch(
        "langgraph_kit.observability.get_config", return_value=_config()
    ):
        assert init_langfuse() is None


def test_create_langfuse_handler_returns_none_when_disabled() -> None:
    with patch(
        "langgraph_kit.observability.get_config", return_value=_config()
    ):
        assert create_langfuse_handler() is None


def test_flush_langfuse_is_noop_when_no_client() -> None:
    obs_mod._langfuse_client = None
    # Must not raise even without a client.
    flush_langfuse()


def test_flush_langfuse_swallows_client_errors() -> None:
    broken = MagicMock()
    broken.flush.side_effect = RuntimeError("transient")
    obs_mod._langfuse_client = broken
    # Should not raise — the helper logs and continues.
    flush_langfuse()
    # Reset to avoid cross-test leakage.
    obs_mod._langfuse_client = None


def test_shutdown_langfuse_clears_client_and_swallows_errors() -> None:
    broken = MagicMock()
    broken.shutdown.side_effect = RuntimeError("shutdown failed")
    obs_mod._langfuse_client = broken
    shutdown_langfuse()
    assert obs_mod._langfuse_client is None


def test_shutdown_langfuse_noop_when_no_client() -> None:
    obs_mod._langfuse_client = None
    shutdown_langfuse()  # Must not raise.


def test_build_agent_run_config_carries_thread_and_user_metadata() -> None:
    with patch(
        "langgraph_kit.observability.get_config",
        return_value=_config(environment="dev"),
    ):
        config = build_agent_run_config(
            agent_id="my-agent",
            thread_id="t-1",
            current_user=_User(),
            endpoint="invoke",
        )

    assert config["configurable"]["thread_id"] == "t-1"
    assert config["run_name"] == "my-agent.invoke"
    assert "my-agent" in config["tags"]
    assert "invoke" in config["tags"]
    metadata = config["metadata"]
    assert metadata["agent_id"] == "my-agent"
    assert metadata["user_id"] == "user-1"
    assert metadata["user_email"] == "u@example.com"
    assert metadata["environment"] == "dev"


def test_build_agent_run_config_attaches_budget_callback_when_enabled() -> None:
    with patch(
        "langgraph_kit.observability.get_config",
        return_value=_config(token_budget_per_thread=10_000),
    ):
        config = build_agent_run_config(
            agent_id="agent",
            thread_id="t",
            current_user=_User(),
            endpoint="stream",
        )

    callbacks = config.get("callbacks", [])
    assert callbacks, "Token-budget tracking should attach a callback"
    assert "_budget_callback" in config["metadata"]
