"""Regression: ``PersistentMemoryManager.get`` must reconcile duplicates.

A crash between the write-new and delete-old steps of ``update()`` when
scope or type changes leaves the record present in two namespaces. Later
``get(...)`` calls previously returned whichever came first in iteration
order — which was the stale copy in the old type's namespace. The
self-heal logic now returns the most-recently-updated record and deletes
the orphan.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

from .conftest import MockStore


@pytest.mark.asyncio
async def test_get_returns_newest_when_duplicate_in_two_type_namespaces() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    record_id = "mem-1"
    scope = MemoryScope.USER

    old = MemoryRecord(
        id=record_id,
        title="Old",
        body="stale content",
        type=MemoryType.FEEDBACK,
        scope=scope,
        summary="stale",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    new = MemoryRecord(
        id=record_id,
        title="New",
        body="winning content",
        type=MemoryType.PROJECT,
        scope=scope,
        summary="fresh",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 4, 1, tzinfo=UTC),
    )

    # Simulate the post-crash state: both records live under their
    # respective (memory, user, TYPE) namespaces.
    await store.aput(
        ("memory", scope.value, old.type.value), record_id, old.to_store_value()
    )
    await store.aput(
        ("memory", scope.value, new.type.value), record_id, new.to_store_value()
    )

    result = await mgr.get(record_id, scope)

    assert result is not None
    assert result.body == "winning content", (
        "get() should return the most recently updated record when "
        "duplicates exist — not whatever came first in iteration order."
    )

    # Orphan is cleaned up opportunistically.
    stale_item = await store.aget(("memory", scope.value, old.type.value), record_id)
    assert stale_item is None, (
        "Stale duplicate in the old namespace should be deleted on read."
    )


@pytest.mark.asyncio
async def test_get_returns_single_record_unchanged() -> None:
    """No duplicate → no self-heal path; normal return."""
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    record = MemoryRecord(
        id="mem-1",
        title="title",
        body="body",
        type=MemoryType.USER,
        scope=MemoryScope.USER,
        summary="s",
    )
    await mgr.create(record)

    out = await mgr.get("mem-1", MemoryScope.USER)
    assert out is not None
    assert out.id == "mem-1"
