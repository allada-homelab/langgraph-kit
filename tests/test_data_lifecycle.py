"""Coverage — DataLifecycleManager export / delete / anonymize.

Issue #31 ships the foundation: per-user lifecycle ops over the
namespaces where ``user_id`` is already present (the per-user thread
index and the actor-keyed audit log). Wider coverage will follow as
#33 multi-tenancy adds tenant scoping to the rest of the namespaces.

The manager always writes an audit entry for the lifecycle event
itself, so a record of "user X requested deletion at time T" lives
on even after their actual data is gone.
"""

from __future__ import annotations

import pytest

from langgraph_kit.core.audit import AuditAction, AuditStore
from langgraph_kit.core.lifecycle import DataLifecycleManager
from langgraph_kit.core.lifecycle.manager import _pseudonym
from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


async def _seed_user_threads(store: MockStore, user_id: str, n: int) -> None:
    ns = ("thread_index", "by_user", user_id)
    for i in range(n):
        await store.aput(
            ns,
            f"thread-{i}",
            {"thread_id": f"thread-{i}", "user_id": user_id, "title": f"chat {i}"},
        )


async def _seed_user_audit(audit: AuditStore, user_id: str, n: int) -> None:
    for i in range(n):
        await audit.write(
            actor=f"user:{user_id}",
            action=AuditAction.AGENT_INVOKE,
            target=f"thread:t-{i}",
            metadata={"index": i},
        )


# ---------------------------------------------------------------------------
# Pseudonym helper
# ---------------------------------------------------------------------------


def test_pseudonym_is_stable_for_same_user_and_salt() -> None:
    a = _pseudonym("alice", "salt-A")
    b = _pseudonym("alice", "salt-A")
    assert a == b


def test_pseudonym_changes_with_salt() -> None:
    assert _pseudonym("alice", "salt-A") != _pseudonym("alice", "salt-B")


def test_pseudonym_changes_with_user_id() -> None:
    assert _pseudonym("alice", "S") != _pseudonym("bob", "S")


def test_pseudonym_starts_with_anon_prefix() -> None:
    assert _pseudonym("u", "s").startswith("anon-")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_returns_threads_and_audit_for_user() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_threads(store, "alice", n=3)
    await _seed_user_audit(audit, "alice", n=2)

    mgr = DataLifecycleManager(store, audit=audit)
    result = await mgr.export("alice")

    assert {item["key"] for item in result["threads"]} == {
        "thread-0",
        "thread-1",
        "thread-2",
    }
    assert len(result["audit"]) == 2
    # The export call itself is an audit event.
    found = await audit.query(action=AuditAction.DATA_EXPORT)
    assert len(found) == 1
    assert found[0].metadata["thread_count"] == 3
    assert found[0].metadata["audit_count"] == 2


@pytest.mark.asyncio
async def test_export_returns_empty_for_unknown_user() -> None:
    store = MockStore()
    audit = AuditStore(store)
    mgr = DataLifecycleManager(store, audit=audit)
    result = await mgr.export("ghost")
    assert result == {"threads": [], "audit": []}


@pytest.mark.asyncio
async def test_export_does_not_include_other_users() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_threads(store, "alice", n=1)
    await _seed_user_threads(store, "bob", n=2)
    await _seed_user_audit(audit, "alice", n=1)
    await _seed_user_audit(audit, "bob", n=1)

    mgr = DataLifecycleManager(store, audit=audit)
    alice = await mgr.export("alice")
    assert all(
        "user_id" in t["value"] and t["value"]["user_id"] == "alice"
        for t in alice["threads"]
    )
    assert all(a["value"]["actor"] == "user:alice" for a in alice["audit"])


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_threads_and_audit_for_user() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_threads(store, "alice", n=2)
    await _seed_user_audit(audit, "alice", n=3)
    # Other user untouched.
    await _seed_user_threads(store, "bob", n=1)

    mgr = DataLifecycleManager(store, audit=audit)
    removed = await mgr.delete("alice")
    assert removed == 5  # 2 threads + 3 audit entries

    # Alice's thread index is now empty.
    follow_up = await mgr.export("alice")
    assert follow_up["threads"] == []
    # The audit list contains only the lifecycle events the manager
    # itself wrote afterwards (DATA_EXPORT / DATA_DELETE). The original
    # AGENT_INVOKE rows are gone.
    actions_seen = {row["value"]["action"] for row in follow_up["audit"]}
    assert AuditAction.AGENT_INVOKE.value not in actions_seen
    survivors = await audit.query(actor="user:alice", action=AuditAction.AGENT_INVOKE)
    assert survivors == []
    # Bob is unaffected.
    bob = await mgr.export("bob")
    assert len(bob["threads"]) == 1


@pytest.mark.asyncio
async def test_delete_writes_audit_event() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_threads(store, "alice", n=1)
    mgr = DataLifecycleManager(store, audit=audit)
    await mgr.delete("alice")
    rows = await audit.query(action=AuditAction.DATA_DELETE)
    assert len(rows) >= 1
    assert rows[0].target == "user:alice"


@pytest.mark.asyncio
async def test_delete_unknown_user_is_noop_returns_zero() -> None:
    store = MockStore()
    audit = AuditStore(store)
    mgr = DataLifecycleManager(store, audit=audit)
    assert await mgr.delete("ghost") == 0


# ---------------------------------------------------------------------------
# anonymize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anonymize_replaces_user_in_thread_index() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_threads(store, "alice", n=2)
    mgr = DataLifecycleManager(store, audit=audit, anonymize_salt="S")

    rewritten = await mgr.anonymize("alice")
    assert rewritten >= 2

    # Original namespace empty, pseudonym namespace populated.
    pseudonym = _pseudonym("alice", "S")
    old_ns = ("thread_index", "by_user", "alice")
    new_ns = ("thread_index", "by_user", pseudonym)
    assert store._data.get(old_ns, {}) == {}
    assert len(store._data.get(new_ns, {})) == 2
    # Each rewritten record has the new user_id.
    for value in store._data[new_ns].values():
        assert value["user_id"] == pseudonym


@pytest.mark.asyncio
async def test_anonymize_rewrites_audit_actor_in_place() -> None:
    store = MockStore()
    audit = AuditStore(store)
    await _seed_user_audit(audit, "alice", n=2)
    mgr = DataLifecycleManager(store, audit=audit, anonymize_salt="S")

    await mgr.anonymize("alice")
    pseudonym = _pseudonym("alice", "S")
    rows = await audit.query(actor=f"user:{pseudonym}")
    assert len(rows) >= 2
    # No actor=user:alice rows remain (other than the anonymize event
    # itself, which we filter out).
    leftover = await audit.query(actor="user:alice", action=AuditAction.AGENT_INVOKE)
    assert leftover == []


# ---------------------------------------------------------------------------
# Audit-store optional
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_works_without_audit_store() -> None:
    """Lifecycle ops must not crash when audit isn't wired (test envs,
    early bring-up). The audit-event side effect is silently skipped."""
    store = MockStore()
    await _seed_user_threads(store, "alice", n=1)
    mgr = DataLifecycleManager(store, audit=None)

    exported = await mgr.export("alice")
    assert len(exported["threads"]) == 1
    assert await mgr.delete("alice") == 1
