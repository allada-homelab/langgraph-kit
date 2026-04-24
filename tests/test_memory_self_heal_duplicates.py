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


@pytest.mark.asyncio
async def test_self_heal_handles_three_or_more_duplicates() -> None:
    """The reconcile loop must return newest and delete *all* older copies."""
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    record_id = "mem-1"
    scope = MemoryScope.USER
    triples = [
        (MemoryType.FEEDBACK, datetime(2026, 1, 1, tzinfo=UTC), "oldest"),
        (MemoryType.PROJECT, datetime(2026, 2, 1, tzinfo=UTC), "middle"),
        (MemoryType.USER, datetime(2026, 3, 1, tzinfo=UTC), "newest"),
    ]
    for mtype, ts, label in triples:
        rec = MemoryRecord(
            id=record_id,
            title=label,
            body=label,
            type=mtype,
            scope=scope,
            summary=label,
            created_at=ts,
            updated_at=ts,
        )
        await store.aput(
            ("memory", scope.value, mtype.value), record_id, rec.to_store_value()
        )

    result = await mgr.get(record_id, scope)
    assert result is not None
    assert result.body == "newest"

    # Every older copy must be gone.
    for mtype, _, label in triples[:-1]:
        leftover = await store.aget(("memory", scope.value, mtype.value), record_id)
        assert leftover is None, (
            f"Stale duplicate {label!r} in {mtype.value} namespace was not "
            "cleaned up on read."
        )


@pytest.mark.asyncio
async def test_self_heal_tolerates_delete_failure() -> None:
    """If opportunistic cleanup fails, ``get`` still returns the winner.

    The store may momentarily reject a delete (transient connection error,
    rate limit, etc.). Self-heal logs the failure and moves on — the
    caller still gets the newest record so work can continue. The next
    read will re-attempt the cleanup.
    """
    store = MockStore()

    original_adelete = store.adelete
    call_count = {"n": 0}

    async def flaky_adelete(namespace: tuple[str, ...], key: str) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated store outage")
        await original_adelete(namespace, key)

    store.adelete = flaky_adelete  # type: ignore[method-assign]

    mgr = PersistentMemoryManager(store)
    scope = MemoryScope.USER
    record_id = "mem-1"
    old = MemoryRecord(
        id=record_id,
        title="old",
        body="old",
        type=MemoryType.FEEDBACK,
        scope=scope,
        summary="old",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    new = MemoryRecord(
        id=record_id,
        title="new",
        body="new",
        type=MemoryType.PROJECT,
        scope=scope,
        summary="new",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    await store.aput(
        ("memory", scope.value, old.type.value), record_id, old.to_store_value()
    )
    await store.aput(
        ("memory", scope.value, new.type.value), record_id, new.to_store_value()
    )

    # First read: delete raises but winner still returned.
    result = await mgr.get(record_id, scope)
    assert result is not None
    assert result.body == "new"

    # Second read: delete now succeeds, stale copy is gone.
    result2 = await mgr.get(record_id, scope)
    assert result2 is not None
    assert result2.body == "new"
    stale = await store.aget(("memory", scope.value, old.type.value), record_id)
    assert stale is None
