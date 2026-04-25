"""Coverage — append-only audit log added by issue #24.

The :class:`AuditStore` is the only sanctioned read/write surface
for audit data. Entries land in time-bucketed namespaces so monthly
listings stay cheap; query supports newest-first iteration with
filters by actor / action / target / time window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from langgraph_kit.core.audit import AuditAction, AuditEntry, AuditStore
from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_audit_entry_defaults_are_set() -> None:
    entry = AuditEntry(
        actor="user:1", action=AuditAction.AGENT_INVOKE, target="thread:abc"
    )
    assert entry.id  # UUID populated
    assert entry.timestamp.tzinfo is not None
    assert entry.metadata == {}


def test_bucket_key_is_year_month() -> None:
    entry = AuditEntry(
        actor="user:1",
        action=AuditAction.AGENT_INVOKE,
        target="thread:abc",
        timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
    )
    assert entry.bucket_key() == "2026_04"


def test_audit_action_enum_has_expected_members() -> None:
    members = {a.value for a in AuditAction}
    assert "agent_invoke" in members
    assert "memory_create" in members
    assert "injection_detected" in members
    assert "output_redacted" in members


# ---------------------------------------------------------------------------
# Store: write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_persists_entry_to_correct_bucket() -> None:
    store = MockStore()
    audit = AuditStore(store)
    entry = await audit.write(
        actor="user:1",
        action=AuditAction.AGENT_INVOKE,
        target="thread:abc",
        metadata={"foo": "bar"},
    )
    bucket_ns = ("audit", entry.bucket_key())
    assert entry.id in store._data.get(bucket_ns, {})
    payload = store._data[bucket_ns][entry.id]
    assert payload["actor"] == "user:1"
    assert payload["action"] == "agent_invoke"
    assert payload["metadata"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_write_returns_persisted_entry_with_id_and_timestamp() -> None:
    audit = AuditStore(MockStore())
    entry = await audit.write(
        actor="system",
        action=AuditAction.OUTPUT_REDACTED,
        target="message:m1",
    )
    assert entry.id
    assert entry.actor == "system"
    assert entry.timestamp <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_write_failure_in_underlying_store_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Audit must never block a real action — Store failures get logged."""

    class _BrokenStore(MockStore):
        async def aput(self, namespace, key, value) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("audit store offline")

    audit = AuditStore(_BrokenStore())
    with caplog.at_level("ERROR"):
        result = await audit.write(
            actor="system",
            action=AuditAction.AGENT_INVOKE,
            target="thread:1",
        )
    assert result.actor == "system"
    assert any("AuditStore.write failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Store: query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_newest_first() -> None:
    audit = AuditStore(MockStore())
    e1 = await audit.write(
        actor="user:1", action=AuditAction.AGENT_INVOKE, target="t:1"
    )
    e2 = await audit.write(
        actor="user:1", action=AuditAction.AGENT_INVOKE, target="t:2"
    )
    e3 = await audit.write(
        actor="user:1", action=AuditAction.AGENT_INVOKE, target="t:3"
    )
    results = await audit.query(limit=10)
    # Within the same bucket, sort is by timestamp descending.
    ids = [r.id for r in results]
    assert ids[0] == e3.id
    assert e2.id in ids
    assert e1.id in ids


@pytest.mark.asyncio
async def test_query_filters_by_actor() -> None:
    audit = AuditStore(MockStore())
    await audit.write(actor="user:1", action=AuditAction.AGENT_INVOKE, target="t1")
    await audit.write(actor="user:2", action=AuditAction.AGENT_INVOKE, target="t2")
    results = await audit.query(actor="user:1")
    assert all(r.actor == "user:1" for r in results)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_query_filters_by_action() -> None:
    audit = AuditStore(MockStore())
    await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target="t1")
    await audit.write(actor="x", action=AuditAction.OUTPUT_REDACTED, target="t1")
    await audit.write(actor="x", action=AuditAction.INJECTION_DETECTED, target="t1")
    results = await audit.query(action=AuditAction.OUTPUT_REDACTED)
    assert len(results) == 1
    assert results[0].action == AuditAction.OUTPUT_REDACTED


@pytest.mark.asyncio
async def test_query_filters_by_target() -> None:
    audit = AuditStore(MockStore())
    await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target="t1")
    await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target="t2")
    results = await audit.query(target="t2")
    assert len(results) == 1
    assert results[0].target == "t2"


@pytest.mark.asyncio
async def test_query_respects_limit() -> None:
    audit = AuditStore(MockStore())
    for i in range(20):
        await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target=f"t:{i}")
    results = await audit.query(limit=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_query_with_zero_limit_returns_empty() -> None:
    audit = AuditStore(MockStore())
    await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target="t")
    assert await audit.query(limit=0) == []


@pytest.mark.asyncio
async def test_query_filters_by_time_window() -> None:
    audit = AuditStore(MockStore())
    await audit.write(actor="x", action=AuditAction.AGENT_INVOKE, target="now")
    now = datetime.now(UTC)
    # Window in the past — excludes the entry we just wrote.
    results = await audit.query(
        since=now - timedelta(days=30), until=now - timedelta(days=29)
    )
    assert results == []
    # Wide window — includes it.
    results = await audit.query(
        since=now - timedelta(hours=1), until=now + timedelta(hours=1)
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_query_walks_buckets_in_reverse_chronological_order() -> None:
    """Spread entries across two month buckets via direct store write
    (bypassing AuditStore.write so we control the timestamp), then
    confirm the newer bucket is read first."""
    store = MockStore()
    # Old entry in 2025_12 bucket.
    old_entry = AuditEntry(
        actor="x",
        action=AuditAction.AGENT_INVOKE,
        target="old",
        timestamp=datetime(2025, 12, 15, 12, 0, tzinfo=UTC),
    )
    await store.aput(
        ("audit", "2025_12"), old_entry.id, old_entry.model_dump(mode="json")
    )
    # New entry in 2026_04 bucket.
    new_entry = AuditEntry(
        actor="x",
        action=AuditAction.AGENT_INVOKE,
        target="new",
        timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
    )
    await store.aput(
        ("audit", "2026_04"), new_entry.id, new_entry.model_dump(mode="json")
    )

    audit = AuditStore(store)
    # Query a window that includes both.
    results = await audit.query(
        since=datetime(2025, 1, 1, tzinfo=UTC),
        until=datetime(2026, 12, 31, tzinfo=UTC),
        limit=10,
    )
    assert len(results) == 2
    assert results[0].target == "new"
    assert results[1].target == "old"
