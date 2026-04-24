"""Checkpointer and store factory for LangGraph persistence.

Yields a ``(checkpointer, store)`` tuple. PostgreSQL is detected by the
URL scheme (``postgresql://``, ``postgres://``, ``postgresql+psycopg://``,
``postgresql+psycopg2://``, ``postgresql+asyncpg://``) — anything else
falls back to SQLite checkpointer + in-memory store.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from langgraph_kit._config import get_config

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Schemes the kit routes through the Postgres adapter. ``postgres://`` is
# the Heroku/legacy form; the ``+driver`` variants just pin the DBAPI and
# don't change the wire protocol — LangGraph's from_conn_string only
# accepts the bare ``postgresql`` scheme, so we normalize down to that.
_POSTGRES_SCHEMES = frozenset(
    {
        "postgresql",
        "postgres",
        "postgresql+psycopg",
        "postgresql+psycopg2",
        "postgresql+asyncpg",
    }
)

# Process-wide flag so the "store is in-memory" warning fires exactly
# once per process instead of spamming logs on every request.
_SQLITE_WARNING_EMITTED = False


def _connection_url() -> str:
    """Normalize the configured database URL for LangGraph's checkpoint drivers."""
    uri = get_config().database_url
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if scheme in _POSTGRES_SCHEMES:
        # LangGraph's Postgres saver only accepts the bare scheme, so
        # strip any ``+driver`` suffix and collapse ``postgres://`` to
        # the canonical form.
        netloc_and_path = uri.split("://", 1)[1] if "://" in uri else uri
        return f"postgresql://{netloc_and_path}"
    return uri


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

        # Loudly warn the first time the InMemoryStore fallback engages
        # so operators notice that Store-backed state (memories, tool
        # results, async tasks, budget) does NOT survive process restarts
        # in this mode. This is a common dev→prod gotcha: dev on SQLite
        # "works", prod on Postgres persists, and the divergence is
        # invisible without this warning.
        global _SQLITE_WARNING_EMITTED
        if not _SQLITE_WARNING_EMITTED:
            logger.warning(
                "Using SQLite checkpointer + IN-MEMORY store "
                "(database_url=%r). Store-backed state (memories, "
                "persisted tool results, async tasks, token budget) "
                "will be lost on restart. Configure a PostgreSQL "
                "database_url for durable storage.",
                pg_url,
            )
            _SQLITE_WARNING_EMITTED = True

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            yield checkpointer, InMemoryStore()
