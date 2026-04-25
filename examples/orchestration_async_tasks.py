"""Orchestration: fire-and-forget async tasks tracked in the Store.

What this shows
---------------
- Starting an :class:`AsyncTask` against a registered graph via
  :meth:`AsyncTaskManager.start`
- Polling task state via :meth:`get` / :meth:`list_tasks`
- The completion / error / cancellation lifecycle (``AsyncTaskStatus``)
- Tasks survive context compaction because they're persisted in the
  Store under ``(async_tasks, parent_thread_id, task_id)``

The same manager backs the agent-callable ``start_async_task`` /
``check_async_task`` tools that long-running deep agents use to spawn
parallel sub-agent investigations.

How to run
----------
    uv run python -m examples.orchestration_async_tasks

Expected output
---------------
    Started task: research-thread (status=running)
    Polled while running: status=running
    After completion: status=success result=A LangGraph deep-agent...
"""

from __future__ import annotations

import asyncio

from examples._lib import (
    answer,
    banner,
    line,
    make_in_memory_persistence,
    patch_build_llm,
    scripted_llm,
)


async def main() -> None:
    banner("orchestration_async_tasks")

    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit.core.orchestration.async_tasks import (
        AsyncTaskManager,
        AsyncTaskStatus,
    )
    from langgraph_kit.graphs.echo_agent import build_graph

    # Build a tiny scripted echo graph for the background task to drive.
    with patch_build_llm(
        scripted_llm([answer("A LangGraph deep-agent toolkit summary.")])
    ):
        checkpointer, store = make_in_memory_persistence()
        graph = build_graph(checkpointer, store)

        manager = AsyncTaskManager(store, parent_thread_id="parent-thread")

        # 1. Start. Returns immediately with status=RUNNING.
        task = await manager.start(
            agent_name="echo-agent",
            description="research-thread",
            graph=graph,
            input_data={"messages": [HumanMessage(content="Summarise the kit.")]},
            config={"configurable": {}},
        )
        line(f"Started task: {task.description} (status={task.status.value})")

        # 2. Poll once. The asyncio task may still be RUNNING because
        # it hasn't yielded yet.
        polled = await manager.check(task.task_id)
        if polled is not None:
            line(f"Polled while running: status={polled.status.value}")

        # 3. Wait for completion. ``manager.check`` is the only public
        # poll — examples wait via the underlying asyncio.Task that
        # ``start()`` records, then poll once more for the persisted state.
        bg = manager._running_asyncio_tasks.get(task.task_id)
        if bg is not None:
            await bg

        # 4. Poll after completion.
        finished = await manager.check(task.task_id)
        if finished is None:
            line("Unexpected: task disappeared from the Store.")
            return
        result_preview = (finished.result or "")[:60]
        line(
            f"After completion: status={finished.status.value} result={result_preview}"
        )

        running_count = len(await manager.list_tasks(AsyncTaskStatus.RUNNING))
        success_count = len(await manager.list_tasks(AsyncTaskStatus.SUCCESS))
        line(f"List by status: running={running_count}  success={success_count}")


if __name__ == "__main__":
    asyncio.run(main())
