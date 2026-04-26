"""End-to-end tests for ``POST /agents/{id}/threads/{tid}/cancel``."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from langgraph_kit.cancellation import get_cancellation_registry
from langgraph_kit.contrib.fastapi import create_agent_router
from langgraph_kit.core.threads import ThreadManager

from .conftest import MockStore


class _User:
    def __init__(self, uid: str) -> None:
        self.id = uid
        self.email = f"{uid}@example.test"


def _make_app(store: Any, user_id: str = "alice") -> FastAPI:
    """Build a FastAPI app wired with the agent router and a fixed user."""

    def _get_user() -> _User:
        return _User(user_id)

    current_user: Any = Annotated[_User, Depends(_get_user)]
    app = FastAPI()
    app.state.store = store
    app.include_router(
        create_agent_router(get_current_user=current_user), prefix="/api/v1"
    )
    return app


class TestCancelEndpoint:
    """The cancel endpoint maps cleanly to the cancellation registry."""

    def test_cancel_no_running_run_returns_false(self) -> None:
        store = MockStore()
        # Pre-claim the thread so the owner check passes.
        asyncio.run(
            ThreadManager(store).ensure_thread(
                thread_id="t-idle",
                user_id="alice",
                agent_id="a1",
                first_message="hi",
            )
        )

        client = TestClient(_make_app(store))
        resp = client.post("/api/v1/agents/a1/threads/t-idle/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"thread_id": "t-idle", "cancelled": False}

    def test_cancel_running_run_returns_true(self) -> None:
        """Register a long-running task on the singleton and cancel it via HTTP."""
        store = MockStore()
        asyncio.run(
            ThreadManager(store).ensure_thread(
                thread_id="t-busy",
                user_id="alice",
                agent_id="a1",
                first_message="hi",
            )
        )

        registry = get_cancellation_registry()
        # Use a fresh event loop to host the task while the TestClient
        # talks to the app via its own loop. That mirrors the
        # multi-task production setup where the cancelling request
        # hits a worker that has the task registered on a different
        # call frame.
        loop = asyncio.new_event_loop()

        async def _runner() -> None:
            await asyncio.sleep(60)

        task = loop.create_task(_runner())
        # Spin once so the task is actually started.
        loop.call_soon(loop.stop)
        loop.run_forever()

        try:
            registry.register("t-busy", task)
            client = TestClient(_make_app(store))
            resp = client.post("/api/v1/agents/a1/threads/t-busy/cancel")
            assert resp.status_code == 200
            assert resp.json() == {"thread_id": "t-busy", "cancelled": True}

            # Drive the loop briefly so the cancellation propagates.
            with suppress(asyncio.CancelledError):
                loop.run_until_complete(asyncio.wait_for(task, timeout=1.0))
            assert task.cancelled() or task.done()
        finally:
            registry.unregister("t-busy")
            if not task.done():
                task.cancel()
            loop.close()

    def test_cancel_other_users_thread_returns_404(self) -> None:
        """Owner check uses the same 404-not-403 pattern as other endpoints."""
        store = MockStore()
        asyncio.run(
            ThreadManager(store).ensure_thread(
                thread_id="t-bobs",
                user_id="bob",
                agent_id="a1",
                first_message="hi",
            )
        )

        # Caller is alice, not bob.
        client = TestClient(_make_app(store, user_id="alice"))
        resp = client.post("/api/v1/agents/a1/threads/t-bobs/cancel")
        assert resp.status_code == 404

    def test_cancel_unclaimed_thread_returns_false(self) -> None:
        """``allow_unclaimed`` semantics: not 404, but ``cancelled=False``.

        A run can register before ``_ensure_thread`` claims it (small
        race window). The cancel endpoint allows this so the user can
        still kill an in-flight start; if no run is registered, it
        idempotently returns ``cancelled=False``.
        """
        store = MockStore()
        # No ensure_thread call — the thread is unclaimed.
        client = TestClient(_make_app(store))
        resp = client.post("/api/v1/agents/a1/threads/t-pristine/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"thread_id": "t-pristine", "cancelled": False}


@pytest.mark.asyncio
async def test_invoke_registers_task_for_cancellation() -> None:
    """The invoke handler wraps its work in ``track`` so a cancel issued
    mid-invoke propagates to the LLM call.

    This test simulates what the actual handler does — exercising the
    real handler requires standing up a graph + LLM, which is beyond
    a fast unit test.
    """
    registry = get_cancellation_registry()
    observed_cancel = asyncio.Event()

    async def _fake_invoke_handler(thread_id: str) -> None:
        try:
            async with registry.track(thread_id):
                await asyncio.sleep(60)  # stand-in for graph.ainvoke
        except asyncio.CancelledError:
            observed_cancel.set()
            raise

    task = asyncio.create_task(_fake_invoke_handler("t-invoke-cancel"))
    # Wait for the handler to enter ``track``.
    for _ in range(5):
        await asyncio.sleep(0)

    try:
        assert registry.cancel("t-invoke-cancel") is True
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed_cancel.is_set()
    finally:
        registry.unregister("t-invoke-cancel")
