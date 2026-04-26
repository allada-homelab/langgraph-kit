"""Per-process cancellation for in-flight agent invocations.

Tracks ``thread_id → asyncio.Task`` for whatever transport happens to
be driving the run (``contrib.fastapi``'s ``/invoke`` or ``/stream``
today; CLI / other transports can plug in the same way). External
callers — typically a ``POST /threads/{tid}/cancel`` endpoint — call
:func:`cancel_thread` to issue ``Task.cancel()`` against the tracked
task.

Single-process scope. Multi-worker deployments (uvicorn ``--workers
N``, k8s replicas) won't see each other's tasks: a cancel issued on
worker A for a run owned by worker B returns ``False`` and is a
no-op. Cross-process cancellation needs a sticky-session router or a
shared signal channel and is intentionally out of scope here.

Cancellation is cooperative: ``Task.cancel()`` raises
``CancelledError`` in the running task's next ``await``. LangGraph
checkpointers save state at superstep boundaries, so a cancelled
thread is generally resumable from the most recent checkpoint —
the partial work of a single in-progress superstep may be lost.
That tradeoff is part of the cooperative-cancellation contract; if
you need at-most-once semantics across cancellation, persist state
yourself before the await that you might be killed at.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class ThreadCancellationRegistry:
    """Maps ``thread_id`` to the ``asyncio.Task`` running that thread.

    Per-process. Construct directly only if you need a separate
    registry instance (tests). Production code uses the module-level
    singleton via :func:`get_registry`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, asyncio.Task[object]] = {}

    def register(self, thread_id: str, task: asyncio.Task[object]) -> None:
        """Associate *task* with *thread_id*.

        Re-registering a thread_id overwrites the prior task. The kit
        doesn't enforce single-flight per thread; multiple concurrent
        invocations of the same thread_id (rare and usually a bug)
        will mean the cancel only reaches the most recent one. A
        warning is logged on overwrite to make the bug visible.
        """
        existing = self._tasks.get(thread_id)
        if existing is not None and not existing.done():
            logger.warning(
                "Thread %r already has a running task; overwriting registration. "
                "Concurrent invocations of the same thread_id are not supported "
                "and the prior task will not be reachable for cancellation.",
                thread_id,
            )
        self._tasks[thread_id] = task

    def unregister(self, thread_id: str) -> None:
        """Drop *thread_id*'s task entry. No-op if absent.

        Idempotent — safe to call from a ``finally`` block whether
        the task succeeded, raised, or was cancelled.
        """
        self._tasks.pop(thread_id, None)

    def cancel(self, thread_id: str) -> bool:
        """Issue ``Task.cancel()`` against *thread_id*'s task.

        Returns ``True`` if a tracked, not-yet-done task was found
        and cancellation was issued; ``False`` if no task is
        registered or the registered task already finished. The
        return value is "did we ask the task to stop," not "is the
        task definitely stopped" — cancellation is cooperative and
        observed asynchronously.
        """
        task = self._tasks.get(thread_id)
        if task is None or task.done():
            return False
        return task.cancel()

    def is_running(self, thread_id: str) -> bool:
        """Return ``True`` when a not-yet-done task is registered for *thread_id*."""
        task = self._tasks.get(thread_id)
        return task is not None and not task.done()

    @contextlib.asynccontextmanager
    async def track(self, thread_id: str) -> AsyncIterator[None]:
        """Async context manager that registers ``current_task`` for the duration.

        Used at transport entry-points::

            async with get_registry().track(thread_id):
                result = await graph.ainvoke(...)

        Falls back to a no-op when called outside an asyncio task
        context (``asyncio.current_task()`` returns ``None``); the
        runtime contract for kit transports is "always inside a
        task," so the fallback exists only to avoid crashing in
        synchronous test setups.
        """
        task = asyncio.current_task()
        if task is None:
            yield
            return
        self.register(thread_id, task)
        try:
            yield
        finally:
            self.unregister(thread_id)


_registry = ThreadCancellationRegistry()


def get_cancellation_registry() -> ThreadCancellationRegistry:
    """Return the process-wide :class:`ThreadCancellationRegistry`.

    All transports share this single instance so a cancel issued by
    one (e.g. an HTTP ``/cancel`` endpoint) reaches a run started by
    another (e.g. a CLI invocation) — provided they're in the same
    process, which is the documented scope.
    """
    return _registry


def cancel_thread(thread_id: str) -> bool:
    """Cancel *thread_id*'s in-flight run via the process-wide registry.

    Convenience wrapper for ``get_registry().cancel(thread_id)``. Returns
    ``True`` when cancellation was issued; ``False`` when no in-flight
    run is tracked for *thread_id* (which can mean "already finished",
    "never started", or "running in a different worker process").
    """
    return _registry.cancel(thread_id)


__all__ = [
    "ThreadCancellationRegistry",
    "cancel_thread",
    "get_cancellation_registry",
]
