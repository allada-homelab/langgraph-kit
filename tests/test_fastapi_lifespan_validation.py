"""Tests for ``create_app_lifespan`` config-validation gating (issue #41).

The lifespan now calls ``validate_config`` at startup and raises
``RuntimeError`` on errors (warnings just log). These tests pin that
contract without spinning up a live DB / Langfuse / MCP stack — we
short-circuit the lifespan via the ``register_agents`` callback (the
validation check runs before that callback).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

import pytest
from fastapi import FastAPI

from langgraph_kit._config import AgentConfig, configure, get_config
from langgraph_kit.contrib.fastapi import create_app_lifespan


@pytest.fixture
def _restore_config() -> Any:
    """Restore the original AgentConfig after each test mutates it."""
    original = get_config()
    yield
    configure(original)


@pytest.mark.usefixtures("_restore_config")
@pytest.mark.asyncio
async def test_lifespan_raises_on_invalid_database_url() -> None:
    """A bad ``database_url`` scheme aborts startup with a readable message."""
    configure(AgentConfig(database_url="mysql://localhost/db"))
    lifespan_factory = create_app_lifespan(register_agents=lambda *_a, **_kw: None)

    with pytest.raises(RuntimeError, match="AgentConfig failed validation"):
        async with lifespan_factory(FastAPI()):
            pass


@pytest.mark.usefixtures("_restore_config")
@pytest.mark.asyncio
async def test_lifespan_raises_on_negative_token_budget() -> None:
    """A negative ``token_budget_per_thread`` aborts startup."""
    configure(AgentConfig(token_budget_per_thread=-1))
    lifespan_factory = create_app_lifespan(register_agents=lambda *_a, **_kw: None)

    with pytest.raises(RuntimeError, match="AgentConfig failed validation"):
        async with lifespan_factory(FastAPI()):
            pass


@pytest.mark.usefixtures("_restore_config")
@pytest.mark.asyncio
async def test_lifespan_logs_warnings_but_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warnings (e.g. half-set Langfuse credentials) log but don't abort startup.

    We let the lifespan attempt the real persistence path, catch any
    downstream failure (since we're not connecting to a real DB), and
    only assert that the warning was emitted before that point.
    """
    configure(AgentConfig(langfuse_public_key="pk_only"))
    lifespan_factory = create_app_lifespan(register_agents=lambda *_a, **_kw: None)

    # Real persistence will eventually try to connect; that's outside
    # the scope of this test. Suppress whatever it raises so we can
    # inspect the warning that fired before it.
    with (
        caplog.at_level(logging.WARNING, logger="langgraph_kit.contrib.fastapi"),
        suppress(Exception),
    ):
        async with lifespan_factory(FastAPI()):
            pass

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("AgentConfig warning:" in w for w in warnings), warnings
