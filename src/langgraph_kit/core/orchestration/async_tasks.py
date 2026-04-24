"""Async sub-agents — fire-and-forget background tasks with Store-backed tracking."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AsyncTaskStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class AsyncTask(BaseModel):
    """Metadata for a tracked background task."""

    task_id: str
    agent_name: str
    description: str
    thread_id: str  # sub-agent's own thread_id
    status: AsyncTaskStatus = AsyncTaskStatus.RUNNING
    created_at: str = ""
    last_checked_at: str | None = None
    result: str | None = None

    def is_terminal(self) -> bool:
        return self.status in {
            AsyncTaskStatus.SUCCESS,
            AsyncTaskStatus.ERROR,
            AsyncTaskStatus.CANCELLED,
        }


# ---------------------------------------------------------------------------
# Store-backed task manager
# ---------------------------------------------------------------------------

_NAMESPACE_PREFIX = "async_tasks"


class AsyncTaskManager:
    """Manages background tasks with persistence via LangGraph Store.

    Tasks are keyed by ``(async_tasks, parent_thread_id, task_id)`` in the
    Store so they survive across context compaction and multiple turns.
    """

    def __init__(
        self,
        store: Any,
        parent_thread_id: str,
        graph_registry: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._parent_tid = parent_thread_id
        self._graph_registry = graph_registry or {}
        self._running_asyncio_tasks: dict[str, asyncio.Task[Any]] = {}

    def _namespace(self) -> tuple[str, str]:
        return (_NAMESPACE_PREFIX, self._parent_tid)

    async def _persist(self, task: AsyncTask) -> None:
        await self._store.aput(
            self._namespace(),
            task.task_id,
            task.model_dump(mode="json"),
        )

    async def _load(self, task_id: str) -> AsyncTask | None:
        item = await self._store.aget(self._namespace(), task_id)
        if item is None:
            return None
        return AsyncTask.model_validate(item.value)

    async def list_tasks(
        self, status_filter: AsyncTaskStatus | None = None
    ) -> list[AsyncTask]:
        """List all tracked tasks, optionally filtered by status."""
        items = await self._store.asearch(self._namespace(), limit=100)
        tasks = [AsyncTask.model_validate(item.value) for item in items]
        if status_filter is not None:
            tasks = [t for t in tasks if t.status == status_filter]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    async def start(
        self,
        agent_name: str,
        description: str,
        graph: Any,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> AsyncTask:
        """Launch a background task and return immediately."""
        sub_tid = str(uuid4())
        task = AsyncTask(
            task_id=str(uuid4()),
            agent_name=agent_name,
            description=description,
            thread_id=sub_tid,
            status=AsyncTaskStatus.RUNNING,
            created_at=datetime.now(UTC).isoformat(),
        )
        await self._persist(task)

        # Build sub-agent config with its own thread_id
        sub_config = {**config, "configurable": {"thread_id": sub_tid}}

        # Launch as a background asyncio task
        asyncio_task = asyncio.create_task(
            self._run_graph(task.task_id, graph, input_data, sub_config)
        )
        self._running_asyncio_tasks[task.task_id] = asyncio_task
        return task

    async def _run_graph(
        self,
        task_id: str,
        graph: Any,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Execute the graph and update task status on completion."""
        try:
            result = await graph.ainvoke(input_data, config=config)
            msgs = result.get("messages") or []
            last_msg = msgs[-1] if msgs else None
            content = (
                last_msg.content
                if last_msg and hasattr(last_msg, "content")
                else str(last_msg)
            )
            task = await self._load(task_id)
            if task and not task.is_terminal():
                task.status = AsyncTaskStatus.SUCCESS
                task.result = content
                await self._persist(task)
        except Exception:
            logger.exception("Async task %s failed", task_id)
            task = await self._load(task_id)
            if task and not task.is_terminal():
                task.status = AsyncTaskStatus.ERROR
                task.result = "Task failed — check logs for details."
                await self._persist(task)
        finally:
            self._running_asyncio_tasks.pop(task_id, None)

    async def check(self, task_id: str) -> AsyncTask | None:
        """Check the current status of a task."""
        task = await self._load(task_id)
        if task is None:
            return None
        task.last_checked_at = datetime.now(UTC).isoformat()
        await self._persist(task)
        return task

    async def cancel(self, task_id: str) -> AsyncTask | None:
        """Cancel a running task."""
        task = await self._load(task_id)
        if task is None:
            return None
        if task.is_terminal():
            return task

        # Cancel the asyncio task if still running
        asyncio_task = self._running_asyncio_tasks.pop(task_id, None)
        if asyncio_task and not asyncio_task.done():
            asyncio_task.cancel()

        task.status = AsyncTaskStatus.CANCELLED
        task.result = "Cancelled by user."
        await self._persist(task)
        return task


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def build_async_task_tools(
    manager: AsyncTaskManager,
    available_graphs: dict[str, Any] | None = None,
) -> list[Any]:
    """Create agent tools for managing async background tasks.

    Returns [start_async_task, check_async_task, cancel_async_task, list_async_tasks].
    """
    _graphs = available_graphs or {}

    async def start_async_task(description: str, worker_type: str) -> str:
        """Launch a long-running task in the background. Returns a task_id.

        IMPORTANT: After starting a task, return control to the user — do NOT
        immediately call check_async_task (that defeats the purpose of async).
        Check tasks later when the user asks or when you need the result.

        Args:
            description: What the task should accomplish
            worker_type: Which worker to use (e.g. "researcher", "implementer")
        """
        graph = _graphs.get(worker_type)
        if graph is None:
            available = list(_graphs.keys()) if _graphs else ["none"]
            return (
                f"Worker type '{worker_type}' not available. "
                f"Available: {', '.join(available)}"
            )

        from langchain_core.messages import (
            HumanMessage,  # pyright: ignore[reportMissingModuleSource]
        )

        input_data = {"messages": [HumanMessage(content=description)]}
        config: dict[str, Any] = {}

        task = await manager.start(
            agent_name=worker_type,
            description=description,
            graph=graph,
            input_data=input_data,
            config=config,
        )
        return (
            f"Background task started.\n"
            f"- task_id: {task.task_id}\n"
            f"- worker: {task.agent_name}\n"
            f"- description: {task.description}\n\n"
            f"Use check_async_task('{task.task_id}') later to get the result."
        )

    async def check_async_task(task_id: str) -> str:
        """Check the status and result of a background task.

        Args:
            task_id: The task_id returned by start_async_task
        """
        task = await manager.check(task_id)
        if task is None:
            return f"No task found with id '{task_id}'."

        lines = [
            f"Task: {task.description}",
            f"Worker: {task.agent_name}",
            f"Status: {task.status.value}",
            f"Started: {task.created_at}",
        ]
        if task.result:
            lines.append(f"\nResult:\n{task.result}")
        elif task.status == AsyncTaskStatus.RUNNING:
            lines.append("\nTask is still running. Check again later.")
        return "\n".join(lines)

    async def cancel_async_task(task_id: str) -> str:
        """Cancel a running background task.

        Args:
            task_id: The task_id to cancel
        """
        task = await manager.cancel(task_id)
        if task is None:
            return f"No task found with id '{task_id}'."
        if task.status == AsyncTaskStatus.CANCELLED:
            return f"Task '{task_id}' has been cancelled."
        return (
            f"Task '{task_id}' is already {task.status.value} and cannot be cancelled."
        )

    async def list_async_tasks(status_filter: str = "") -> str:
        """List all background tasks, optionally filtered by status.

        Args:
            status_filter: Optional filter — "running", "success", "error", "cancelled", or "" for all
        """
        filt = None
        if status_filter:
            try:
                filt = AsyncTaskStatus(status_filter.lower())
            except ValueError:
                return f"Invalid status filter '{status_filter}'. Use: running, success, error, cancelled"

        tasks = await manager.list_tasks(status_filter=filt)
        if not tasks:
            label = f" with status '{status_filter}'" if status_filter else ""
            return f"No background tasks found{label}."

        lines = [f"Found {len(tasks)} task(s):\n"]
        for t in tasks:
            lines.append(
                f"- [{t.status.value.upper()}] {t.description} "
                + f"(id: {t.task_id}, worker: {t.agent_name})"
            )
        return "\n".join(lines)

    return [start_async_task, check_async_task, cancel_async_task, list_async_tasks]
