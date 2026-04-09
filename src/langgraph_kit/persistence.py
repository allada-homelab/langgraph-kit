"""Checkpointer and store factory for LangGraph persistence.

Yields a ``(checkpointer, store)`` tuple.  When PostgreSQL is available
(the ``DATABASE_URL`` starts with ``postgresql``), both use Postgres.
Otherwise, falls back to SQLite checkpointer + in-memory store.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from langgraph_kit._config import get_config

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _connection_url() -> str:
    """Normalize the configured database URL for LangGraph's checkpoint drivers."""
    uri = get_config().database_url
    # LangGraph's from_conn_string expects a plain postgresql:// scheme
    return uri.replace("postgresql+psycopg", "postgresql")


@asynccontextmanager
async def create_persistence() -> AsyncGenerator[tuple[Any, Any]]:
    """Async context manager that yields ``(checkpointer, store)``."""
    pg_url = _connection_url()

    if pg_url.startswith("postgresql"):
        from langgraph.checkpoint.postgres.aio import (
            AsyncPostgresSaver,  # pyright: ignore[reportMissingModuleSource]
        )
        from langgraph.store.postgres import (
            AsyncPostgresStore,  # pyright: ignore[reportMissingModuleSource]
        )

        async with (
            AsyncPostgresSaver.from_conn_string(pg_url) as checkpointer,
            AsyncPostgresStore.from_conn_string(pg_url) as store,
        ):
            await checkpointer.setup()
            await store.setup()
            yield checkpointer, store
    else:
        from langgraph.checkpoint.sqlite.aio import (
            AsyncSqliteSaver,  # pyright: ignore[reportMissingModuleSource]
        )
        from langgraph.store.memory import (
            InMemoryStore,  # pyright: ignore[reportMissingModuleSource]
        )

        # Extract filename from sqlite:///path — default to checkpoints.db
        db_path = pg_url.removeprefix("sqlite:///") or "checkpoints.db"
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            yield checkpointer, InMemoryStore()
