"""Regression tests for the async-task manager bug fixes.

Covers:
- ``start_async_task`` inherits tracing/callbacks from the parent config.
- ``check`` is a pure read (no Store write) — fixes the lost-update race.
- Errors captured into ``task.result`` are actionable, not a generic
  "check logs" string.
- ``list_tasks`` / ``list_async_tasks`` accept a ``limit`` argument.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.orchestration.async_tasks import (
    AsyncTask,
    AsyncTaskManager,
    AsyncTaskStatus,
    build_async_task_tools,
)

from .conftest import MockStore


class _FailingGraph:
    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        raise ValueError("synthetic failure for regression")


class _CapturingGraph:
    """Records whatever config it is invoked with, then returns a single AIMsg."""

    def __init__(self) -> None:
        self.captured_config: dict[str, Any] | None = None

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.captured_config = config
        from langchain_core.messages import (
            AIMessage,  # pyright: ignore[reportMissingModuleSource]
        )

        return {"messages": [AIMessage(content="done")]}


@pytest.mark.asyncio
async def test_check_does_not_write_on_read() -> None:
    store = MockStore()
    mgr = AsyncTaskManager(store, parent_thread_id="p1")

    task = AsyncTask(
        task_id="t1",
        agent_name="w",
        description="d",
        thread_id="s1",
        status=AsyncTaskStatus.RUNNING,
        created_at="2026-04-24T00:00:00Z",
    )
    await mgr._persist(task)

    # Record the stored blob exactly as written.
    before = store._data[("async_tasks", "p1")]["t1"].copy()

    await mgr.check("t1")
    await mgr.check("t1")
    await mgr.check("t1")

    after = store._data[("async_tasks", "p1")]["t1"]
    assert after == before, (
        "check() is a read — must not mutate the stored record. "
        "The prior implementation stamped last_checked_at every call, "
        "which raced concurrent checks."
    )


@pytest.mark.asyncio
async def test_run_graph_captures_exception_detail(monkeypatch: Any) -> None:
    store = MockStore()
    mgr = AsyncTaskManager(store, parent_thread_id="p1")

    graph = _FailingGraph()
    task = await mgr.start(
        agent_name="w",
        description="d",
        graph=graph,
        input_data={"messages": []},
        config={},
    )

    # Let the background task run to completion.
    asyncio_task = mgr._running_asyncio_tasks.get(task.task_id)
    if asyncio_task is not None:
        await asyncio_task

    stored = await mgr._load(task.task_id)
    assert stored is not None
    assert stored.status == AsyncTaskStatus.ERROR
    assert stored.result is not None
    assert "ValueError" in stored.result
    assert "synthetic failure for regression" in stored.result


@pytest.mark.asyncio
async def test_list_tasks_honors_limit() -> None:
    store = MockStore()
    mgr = AsyncTaskManager(store, parent_thread_id="p1")

    for i in range(10):
        await mgr._persist(
            AsyncTask(
                task_id=f"t{i}",
                agent_name="w",
                description=f"d{i}",
                thread_id=f"s{i}",
                status=AsyncTaskStatus.SUCCESS,
                created_at=f"2026-04-24T00:00:{i:02d}Z",
            )
        )

    out = await mgr.list_tasks(limit=3)
    assert len(out) == 3


@pytest.mark.asyncio
async def test_list_async_tasks_tool_passes_limit_through() -> None:
    store = MockStore()
    mgr = AsyncTaskManager(store, parent_thread_id="p1")

    for i in range(8):
        await mgr._persist(
            AsyncTask(
                task_id=f"t{i}",
                agent_name="w",
                description=f"d{i}",
                thread_id=f"s{i}",
                status=AsyncTaskStatus.SUCCESS,
                created_at=f"2026-04-24T00:00:{i:02d}Z",
            )
        )

    tools: list[Any] = build_async_task_tools(manager=mgr, available_graphs={})
    list_tool = None
    for t in tools:
        if getattr(t, "__name__", "") == "list_async_tasks":
            list_tool = t
            break
    assert list_tool is not None

    out = await list_tool(status_filter="", limit=2)
    # Each listed task prints on one line with its id; count occurrences.
    hits = sum(1 for line in out.splitlines() if "id: t" in line)
    assert hits == 2


@pytest.mark.asyncio
async def test_start_async_task_inherits_parent_config(monkeypatch: Any) -> None:
    """start_async_task should inherit tracing-relevant config keys."""
    store = MockStore()
    mgr = AsyncTaskManager(store, parent_thread_id="p1")

    captured = _CapturingGraph()
    graphs = {"worker": captured}

    tools: list[Any] = build_async_task_tools(manager=mgr, available_graphs=graphs)
    start_tool = None
    for t in tools:
        if getattr(t, "__name__", "") == "start_async_task":
            start_tool = t
            break
    assert start_tool is not None

    parent_config = {
        "callbacks": ["fake-langfuse-handler"],
        "tags": ["agent:parent"],
        "metadata": {"_trace_handler": "something"},
        "recursion_limit": 50,
        "configurable": {"thread_id": "parent-thread"},
    }
    monkeypatch.setattr(
        "langgraph.config.get_config",
        lambda: parent_config,
    )

    await start_tool(description="do thing", worker_type="worker")

    # Drain the in-flight asyncio task so captured_config is populated.
    for asyncio_task in list(mgr._running_asyncio_tasks.values()):
        await asyncio_task

    cfg = captured.captured_config or {}
    assert cfg.get("callbacks") == ["fake-langfuse-handler"]
    assert cfg.get("tags") == ["agent:parent"]
    assert cfg.get("metadata") == {"_trace_handler": "something"}
    assert cfg.get("recursion_limit") == 50
    # Thread id is overridden for the sub-agent, not inherited.
    assert cfg.get("configurable", {}).get("thread_id") != "parent-thread"
