"""Regression: persistence._connection_url URL scheme whitelist + SQLite warning."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from langgraph_kit._config import AgentConfig, configure
from langgraph_kit.persistence import _connection_url


@pytest.fixture(autouse=True)
def _restore_config() -> Any:
    original = AgentConfig()
    yield
    configure(original)


@pytest.mark.parametrize(
    "input_url",
    [
        "postgresql://user:pw@host/db",
        "postgres://user:pw@host/db",
        "postgresql+psycopg://user:pw@host/db",
        "postgresql+psycopg2://user:pw@host/db",
        "postgresql+asyncpg://user:pw@host/db",
    ],
)
def test_connection_url_normalizes_postgres_variants_to_bare_scheme(
    input_url: str,
) -> None:
    configure(AgentConfig(database_url=input_url))
    normalized = _connection_url()
    assert normalized.startswith("postgresql://"), (
        f"expected postgresql:// scheme; got {normalized!r}"
    )
    assert "+psycopg" not in normalized
    assert "+asyncpg" not in normalized
    assert "+psycopg2" not in normalized


def test_connection_url_passes_sqlite_through_unchanged() -> None:
    configure(AgentConfig(database_url="sqlite:///local.db"))
    assert _connection_url() == "sqlite:///local.db"


@pytest.mark.asyncio
async def test_create_persistence_logs_sqlite_warning_on_fallback(
    caplog: Any,
) -> None:
    from langgraph_kit import persistence as persistence_mod

    # Reset the once-per-process flag so the warning can fire in this test.
    persistence_mod._SQLITE_WARNING_EMITTED = False

    configure(AgentConfig(database_url="sqlite:///:memory:"))
    with caplog.at_level(logging.WARNING, logger="langgraph_kit.persistence"):
        async with persistence_mod.create_persistence() as (_ckpt, _store):
            pass

    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "IN-MEMORY store" in m and "Configure a PostgreSQL" in m for m in messages
    ), f"Expected warning about in-memory store; got: {messages!r}"
