"""Regression tests for FastAPI thread-ownership authorization.

Before this fix, execution endpoints (stream, invoke, resume, fork, queue,
history, messages, state) accepted any authenticated user's request for
any thread_id — a cross-tenant data-access bug. These tests assert that
``_verify_thread_owner`` rejects mismatched owners with 404 (not 403, to
avoid leaking thread existence) and allows unclaimed threads only when
explicitly opted in.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from langgraph_kit.contrib.fastapi import _verify_thread_owner
from langgraph_kit.core.threads import ThreadManager

from .conftest import MockStore


class _User:
    def __init__(self, uid: str) -> None:
        self.id = uid


@pytest.mark.asyncio
async def test_verify_owner_rejects_other_users_thread() -> None:
    store = MockStore()
    mgr = ThreadManager(store)
    await mgr.ensure_thread(
        thread_id="t-owned",
        user_id="alice",
        agent_id="agent-a",
        first_message="hi",
    )

    with pytest.raises(HTTPException) as exc:
        await _verify_thread_owner(store, "t-owned", _User("bob"))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_verify_owner_allows_matching_user() -> None:
    store = MockStore()
    mgr = ThreadManager(store)
    await mgr.ensure_thread(
        thread_id="t-owned",
        user_id="alice",
        agent_id="agent-a",
        first_message="hi",
    )

    # Should not raise.
    await _verify_thread_owner(store, "t-owned", _User("alice"))


@pytest.mark.asyncio
async def test_verify_owner_rejects_unclaimed_thread_by_default() -> None:
    store = MockStore()

    with pytest.raises(HTTPException) as exc:
        await _verify_thread_owner(store, "t-unknown", _User("alice"))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_verify_owner_allows_unclaimed_when_flag_set() -> None:
    """The ``allow_unclaimed=True`` path is used by stream/invoke where the
    first request creates the thread."""
    store = MockStore()

    # Should not raise.
    await _verify_thread_owner(
        store, "t-unknown", _User("alice"), allow_unclaimed=True
    )


@pytest.mark.asyncio
async def test_verify_owner_uses_404_not_403_to_avoid_enumeration() -> None:
    """Returning 403 would reveal that the thread exists. 404 hides it."""
    store = MockStore()
    mgr = ThreadManager(store)
    await mgr.ensure_thread(
        thread_id="t-private",
        user_id="alice",
        agent_id="agent-a",
    )

    # An attacker probing for a valid thread_id gets identical responses
    # for "thread doesn't exist" and "thread belongs to someone else".
    with pytest.raises(HTTPException) as exc1:
        await _verify_thread_owner(store, "t-private", _User("bob"))
    with pytest.raises(HTTPException) as exc2:
        await _verify_thread_owner(store, "t-missing", _User("bob"))
    assert exc1.value.status_code == 404
    assert exc2.value.status_code == 404
    assert exc1.value.detail == exc2.value.detail


@pytest.mark.asyncio
async def test_verify_owner_coerces_user_id_to_string() -> None:
    """``current_user.id`` can be int/UUID; the stored ``user_id`` is str."""
    store = MockStore()
    mgr = ThreadManager(store)
    await mgr.ensure_thread(
        thread_id="t-owned",
        user_id="42",
        agent_id="agent-a",
    )

    class _IntUser:
        id: Any = 42

    # Should not raise — helper calls str(current_user.id).
    await _verify_thread_owner(store, "t-owned", _IntUser())
