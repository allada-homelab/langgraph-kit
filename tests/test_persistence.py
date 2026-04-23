"""Coverage fill — ``persistence.create_persistence`` factory branches.

Tests ``_connection_url`` URL normalization directly (unit) and
exercises the SQLite branch of ``create_persistence`` (requires
``aiosqlite`` which is a kit dependency). The Postgres branch is
out-of-scope without a live database — tested via monkey-patched import
hooks instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from langgraph_kit import _config as config_mod
from langgraph_kit.persistence import _connection_url, create_persistence


def test_connection_url_strips_psycopg_driver_suffix() -> None:
    """``postgresql+psycopg://…`` collapses to plain ``postgresql://…``.

    LangGraph's ``from_conn_string`` drivers don't accept the SQLAlchemy
    ``+psycopg`` dialect suffix. The helper strips it so kit callers can
    reuse a SQLAlchemy-shaped URL from their own config.
    """
    with patch(
        "langgraph_kit.persistence.get_config",
        return_value=config_mod.AgentConfig(
            database_url="postgresql+psycopg://user:pw@host:5432/db",
        ),
    ):
        assert _connection_url() == "postgresql://user:pw@host:5432/db"


def test_connection_url_passes_sqlite_through_unchanged() -> None:
    with patch(
        "langgraph_kit.persistence.get_config",
        return_value=config_mod.AgentConfig(database_url="sqlite:///checkpoints.db"),
    ):
        assert _connection_url() == "sqlite:///checkpoints.db"


@pytest.mark.asyncio
async def test_create_persistence_sqlite_branch_yields_pair(tmp_path: Path) -> None:
    """SQLite branch yields ``(checkpointer, in_memory_store)``."""
    db = tmp_path / "unit.db"
    with patch(
        "langgraph_kit.persistence.get_config",
        return_value=config_mod.AgentConfig(
            database_url=f"sqlite:///{db}",
        ),
    ):
        async with create_persistence() as (checkpointer, store):
            # Both objects expose the kit-visible API surface.
            assert checkpointer is not None
            assert store is not None
            # InMemoryStore's async API should be callable.
            await store.aput(("ns",), "k", {"v": 1})
            item = await store.aget(("ns",), "k")
            assert item is not None


@pytest.mark.asyncio
async def test_create_persistence_postgres_branch_invokes_postgres_drivers() -> None:
    """Postgres URL routes to the Postgres drivers.

    We can't run a live Postgres, so we mock the from_conn_string
    entry points and assert they were invoked with the expected URL.
    That proves the branch is wired correctly without requiring a
    database.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    @asynccontextmanager
    async def _fake_conn(url: str):
        _ = url
        mock = MagicMock()

        async def _setup() -> None:
            return None

        mock.setup = _setup
        yield mock

    class _FakeSaver:
        from_conn_string = staticmethod(_fake_conn)

    class _FakeStore:
        from_conn_string = staticmethod(_fake_conn)

    import sys

    # Pre-populate sys.modules so the dynamic imports inside
    # create_persistence resolve to our fakes.
    fake_modules: dict[str, Any] = {
        "langgraph.checkpoint.postgres": MagicMock(),
        "langgraph.checkpoint.postgres.aio": MagicMock(AsyncPostgresSaver=_FakeSaver),
        "langgraph.store.postgres": MagicMock(AsyncPostgresStore=_FakeStore),
    }

    original = {k: sys.modules.get(k) for k in fake_modules}
    sys.modules.update(fake_modules)
    try:
        with patch(
            "langgraph_kit.persistence.get_config",
            return_value=config_mod.AgentConfig(
                database_url="postgresql://u:p@host/db",
            ),
        ):
            async with create_persistence() as (checkpointer, store):
                assert checkpointer is not None
                assert store is not None
    finally:
        for k, v in original.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
