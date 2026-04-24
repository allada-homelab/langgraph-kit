"""Coverage — graceful shutdown drain for async sub-agent tasks.

Exercises :func:`drain_background_tasks` end-to-end:

- no-op when no tasks are in flight
- tasks that finish inside the window are drained cleanly
- tasks exceeding the window are cancelled, and their Store records
  get updated from RUNNING to CANCELLED so later readers see the real
  terminal state instead of a stuck row
- ``timeout=0`` cancels everything immediately
- the module-level task set auto-cleans via ``done_callback`` so a
  natural-completion flow does not leak entries into subsequent drains
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgraph_kit.core.orchestration.async_tasks import (
    AsyncTaskManager,
    AsyncTaskStatus,
    _background_tasks,
    drain_background_tasks,
)
from tests.conftest import MockStore


class _DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeGraph:
    """Minimal graph driven by AsyncTaskManager; optional gate keeps it running."""

    def __init__(
        self,
        *,
        gate: asyncio.Event | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._gate = gate
        self._result = result or {"messages": [_DummyMessage("done")]}

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        _ = input_data
        _ = config
        if self._gate is not None:
            await self._gate.wait()
        return self._result


@pytest.fixture
def store() -> MockStore:
    return MockStore()


@pytest.fixture
def manager(store: MockStore) -> AsyncTaskManager:
    return AsyncTaskManager(store=store, parent_thread_id="parent-thread")


@pytest.fixture(autouse=True)
def _isolate_module_task_set() -> Any:
    """Clear the module-level background-task set around each test.

    Other test files may (accidentally) leave tasks behind; draining those
    would be slow and unrelated. Snapshot + restore so drain sees only the
    tasks created in this test.
    """
    snapshot = set(_background_tasks)
    _background_tasks.clear()
    yield
    _background_tasks.clear()
    _background_tasks.update(snapshot)


async def test_drain_with_no_tasks_is_noop() -> None:
    drained, cancelled = await drain_background_tasks(timeout=5.0)
    assert (drained, cancelled) == (0, 0)


async def test_drain_completes_quickly_finishing_tasks(
    manager: AsyncTaskManager,
) -> None:
    # No gate → graph returns immediately after event-loop handoff.
    task = await manager.start(
        agent_name="quick",
        description="fast job",
        graph=_FakeGraph(),
        input_data={"messages": []},
        config={},
    )

    drained, cancelled = await drain_background_tasks(timeout=5.0)
    assert drained == 1
    assert cancelled == 0

    record = await manager.check(task.task_id)
    assert record is not None
    assert record.status == AsyncTaskStatus.SUCCESS


async def test_drain_cancels_tasks_exceeding_timeout(
    manager: AsyncTaskManager,
) -> None:
    gate = asyncio.Event()  # never set → graph blocks forever
    task = await manager.start(
        agent_name="slow",
        description="hung job",
        graph=_FakeGraph(gate=gate),
        input_data={"messages": []},
        config={},
    )

    drained, cancelled = await drain_background_tasks(timeout=0.05)
    assert drained == 0
    assert cancelled == 1

    # Store record must reflect the cancellation — otherwise a later
    # reader sees RUNNING forever and would never know the task exited.
    record = await manager.check(task.task_id)
    assert record is not None
    assert record.status == AsyncTaskStatus.CANCELLED


async def test_drain_with_zero_timeout_cancels_immediately(
    manager: AsyncTaskManager,
) -> None:
    gate = asyncio.Event()
    task = await manager.start(
        agent_name="zero",
        description="immediate-cancel job",
        graph=_FakeGraph(gate=gate),
        input_data={"messages": []},
        config={},
    )

    drained, cancelled = await drain_background_tasks(timeout=0.0)
    assert drained == 0
    assert cancelled == 1

    record = await manager.check(task.task_id)
    assert record is not None
    assert record.status == AsyncTaskStatus.CANCELLED


async def test_module_set_auto_cleans_after_natural_completion(
    manager: AsyncTaskManager,
) -> None:
    _ = await manager.start(
        agent_name="clean",
        description="auto-clean job",
        graph=_FakeGraph(),
        input_data={"messages": []},
        config={},
    )
    # Drain to force the task to complete and the done_callback to fire.
    await drain_background_tasks(timeout=5.0)
    assert len(_background_tasks) == 0


async def test_drain_handles_mixed_fast_and_slow(
    manager: AsyncTaskManager,
) -> None:
    fast_task = await manager.start(
        agent_name="fast",
        description="finishes in time",
        graph=_FakeGraph(),
        input_data={"messages": []},
        config={},
    )
    slow_gate = asyncio.Event()
    slow_task = await manager.start(
        agent_name="slow",
        description="does not finish",
        graph=_FakeGraph(gate=slow_gate),
        input_data={"messages": []},
        config={},
    )

    drained, cancelled = await drain_background_tasks(timeout=0.2)
    assert drained == 1
    assert cancelled == 1

    fast_rec = await manager.check(fast_task.task_id)
    slow_rec = await manager.check(slow_task.task_id)
    assert fast_rec is not None
    assert fast_rec.status == AsyncTaskStatus.SUCCESS
    assert slow_rec is not None
    assert slow_rec.status == AsyncTaskStatus.CANCELLED
