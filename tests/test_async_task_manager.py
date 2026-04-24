"""Coverage fill — ``AsyncTaskManager`` full lifecycle.

Exercises the real manager (store-backed) through a fake graph so the
``start → check → list → cancel`` lifecycle is driven end-to-end
without needing a real LangGraph. Fills the 40+% of
``async_tasks.py`` that's uncovered because the existing e2e tests
only touch the empty-config error paths.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgraph_kit.core.orchestration.async_tasks import (
    AsyncTaskManager,
    AsyncTaskStatus,
    build_async_task_tools,
)
from tests.conftest import MockStore


class _FakeGraph:
    """Minimal fake graph for AsyncTaskManager to drive.

    ``ainvoke`` either returns a canned result or awaits a provided
    asyncio.Event so the test can keep the task "running" for a while.
    """

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        gate: asyncio.Event | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        super().__init__()
        self._result = result or {"messages": [_DummyMessage("done")]}
        self._gate = gate
        self._raise = raise_exc

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        _ = input_data
        _ = config
        if self._gate is not None:
            await self._gate.wait()
        if self._raise is not None:
            raise self._raise
        return self._result


class _DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
def store() -> MockStore:
    return MockStore()


@pytest.fixture
def manager(store: MockStore) -> AsyncTaskManager:
    return AsyncTaskManager(store=store, parent_thread_id="parent-thread")


@pytest.mark.asyncio
async def test_start_persists_running_task_and_returns_immediately(
    manager: AsyncTaskManager,
) -> None:
    gate = asyncio.Event()
    graph = _FakeGraph(gate=gate)

    task = await manager.start(
        agent_name="worker-a",
        description="long job",
        graph=graph,
        input_data={"messages": []},
        config={},
    )
    assert task.status == AsyncTaskStatus.RUNNING

    # A ``check`` before the gate opens sees the task as still running.
    # (check() is a pure read now — no last_checked_at stamp.)
    checked = await manager.check(task.task_id)
    assert checked is not None
    assert checked.status == AsyncTaskStatus.RUNNING

    # Release the gate and let the background task complete.
    gate.set()
    # Give the event loop a tick.
    await asyncio.sleep(0.05)

    after = await manager.check(task.task_id)
    assert after is not None
    assert after.status == AsyncTaskStatus.SUCCESS
    assert after.result == "done"


@pytest.mark.asyncio
async def test_failed_background_task_records_error_status(
    manager: AsyncTaskManager,
) -> None:
    graph = _FakeGraph(raise_exc=RuntimeError("boom"))

    task = await manager.start(
        agent_name="failing",
        description="will fail",
        graph=graph,
        input_data={"messages": []},
        config={},
    )
    await asyncio.sleep(0.05)

    after = await manager.check(task.task_id)
    assert after is not None
    assert after.status == AsyncTaskStatus.ERROR
    assert after.result is not None
    assert "failed" in after.result.lower()


@pytest.mark.asyncio
async def test_cancel_running_task_marks_cancelled(
    manager: AsyncTaskManager,
) -> None:
    gate = asyncio.Event()
    graph = _FakeGraph(gate=gate)

    task = await manager.start(
        agent_name="cancellable",
        description="will be cancelled",
        graph=graph,
        input_data={"messages": []},
        config={},
    )
    cancelled = await manager.cancel(task.task_id)
    assert cancelled is not None
    assert cancelled.status == AsyncTaskStatus.CANCELLED

    # Let the event loop unwind any pending cancellation.
    gate.set()
    await asyncio.sleep(0.05)

    again = await manager.check(task.task_id)
    assert again is not None
    # Cancel is terminal — the _run_graph's finally block shouldn't have
    # overwritten the terminal status.
    assert again.status == AsyncTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_nonexistent_task_returns_none(
    manager: AsyncTaskManager,
) -> None:
    assert await manager.cancel("no-such-task") is None


@pytest.mark.asyncio
async def test_cancel_already_terminal_task_is_idempotent(
    manager: AsyncTaskManager,
) -> None:
    graph = _FakeGraph()
    task = await manager.start(
        agent_name="short",
        description="quick",
        graph=graph,
        input_data={"messages": []},
        config={},
    )
    # Let it finish.
    await asyncio.sleep(0.05)
    after = await manager.check(task.task_id)
    assert after is not None
    assert after.status == AsyncTaskStatus.SUCCESS

    # Cancel after success returns the existing terminal task unchanged.
    cancelled = await manager.cancel(task.task_id)
    assert cancelled is not None
    assert cancelled.status == AsyncTaskStatus.SUCCESS


@pytest.mark.asyncio
async def test_list_tasks_filter_by_status(
    manager: AsyncTaskManager,
) -> None:
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    await manager.start(
        agent_name="a",
        description="a-desc",
        graph=_FakeGraph(gate=gate_a),
        input_data={},
        config={},
    )
    task_b = await manager.start(
        agent_name="b",
        description="b-desc",
        graph=_FakeGraph(gate=gate_b),
        input_data={},
        config={},
    )
    # Cancel b to get a terminal task.
    await manager.cancel(task_b.task_id)
    gate_b.set()

    running_only = await manager.list_tasks(status_filter=AsyncTaskStatus.RUNNING)
    assert [t.agent_name for t in running_only] == ["a"]

    cancelled_only = await manager.list_tasks(status_filter=AsyncTaskStatus.CANCELLED)
    assert [t.agent_name for t in cancelled_only] == ["b"]

    # Cleanup — release the running task.
    gate_a.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tool_surface_start_check_cancel_list_round_trip(
    manager: AsyncTaskManager,
) -> None:
    """``build_async_task_tools(manager, available_graphs=...)`` happy path."""
    gate = asyncio.Event()
    graph = _FakeGraph(gate=gate, result={"messages": [_DummyMessage("fetched")]})
    tools = build_async_task_tools(manager, available_graphs={"researcher": graph})
    start, check, cancel, list_ = tools

    started = await start(description="dig logs", worker_type="researcher")
    # Tool surfaces a task_id so subsequent tools can reference it.
    assert "task_id:" in started
    task_id = started.split("task_id:")[1].split("\n")[0].strip()

    # check_async_task returns a running status while the gate is set.
    running = await check(task_id)
    assert "running" in running.lower()

    # list_async_tasks with no filter includes this task.
    listing_all = await list_("")
    assert "dig logs" in listing_all

    # Status filter for success currently returns none.
    listing_success = await list_("success")
    assert "No background tasks" in listing_success

    # Finish the task.
    gate.set()
    await asyncio.sleep(0.05)
    done = await check(task_id)
    assert "success" in done.lower()
    assert "fetched" in done

    # cancel_async_task on a terminal task returns a graceful message.
    cancel_msg = await cancel(task_id)
    assert "already" in cancel_msg.lower()
