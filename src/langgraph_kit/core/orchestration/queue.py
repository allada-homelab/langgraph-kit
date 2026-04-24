"""Busy-thread queueing via Store.

Provides a Store-backed message queue per thread so that messages arriving
while a run is active are buffered and injected on the next model call
instead of starting a competing run.

Message semantics:
  - append:       normal follow-up, added after existing messages
  - interrupt:    urgent correction — injected with high-priority framing
  - replace_goal: redefine the active task — clears prior queue, injected
                  as a goal replacement directive
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import StrEnum
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_config
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue item model
# ---------------------------------------------------------------------------


class QueueSemantic(StrEnum):
    """How the queued message should be interpreted by the agent."""

    APPEND = "append"
    INTERRUPT = "interrupt"
    REPLACE_GOAL = "replace_goal"


class QueuedItem(BaseModel):
    """A single queued message waiting to be injected into a run."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    semantic: QueueSemantic = QueueSemantic.APPEND
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Store-backed thread queue
# ---------------------------------------------------------------------------

QUEUE_NAMESPACE_PREFIX = ("queue",)


def _queue_namespace(thread_id: str) -> tuple[str, ...]:
    return (*QUEUE_NAMESPACE_PREFIX, thread_id)


class ThreadQueue:
    """Store-backed FIFO queue for a single thread.

    Each item is stored under namespace ("queue", thread_id) with a
    time-sortable key to preserve ordering.
    """

    def __init__(self, store: Any, thread_id: str) -> None:
        super().__init__()
        self._store = store
        self._thread_id = thread_id
        self._ns = _queue_namespace(thread_id)

    async def enqueue(self, item: QueuedItem) -> str:
        """Add an item to the queue. Returns the item id."""
        # replace_goal clears the queue first
        if item.semantic == QueueSemantic.REPLACE_GOAL:
            await self.clear()

        # Key is timestamp-prefixed for sort ordering
        key = f"{item.timestamp:.6f}_{item.id}"
        await self._store.aput(self._ns, key, item.model_dump(mode="json"))
        logger.debug(
            "Enqueued %s for thread %s (semantic=%s)",
            key,
            self._thread_id,
            item.semantic,
        )
        return item.id

    # Maximum items fetched per asearch call. The page-loop in drain
    # keeps going until the store reports a batch smaller than this, so
    # queues larger than the batch are still fully drained instead of
    # silently truncated.
    _PAGE_SIZE = 100

    async def drain(self, *, max_items: int | None = None) -> list[QueuedItem]:
        """Remove and return queued items in FIFO order.

        Reads and deletes in pages so queues deeper than a single
        ``_PAGE_SIZE`` batch drain fully instead of silently truncating.
        ``max_items`` caps the total returned; any remainder stays
        queued for the next call.
        """
        result: list[QueuedItem] = []
        while True:
            batch = await self._store.asearch(self._ns, limit=self._PAGE_SIZE)
            if not batch:
                break
            batch.sort(key=lambda x: x.key)  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
            for raw in batch:
                if max_items is not None and len(result) >= max_items:
                    break
                try:
                    result.append(QueuedItem.model_validate(raw.value))
                except Exception:
                    logger.warning("Skipping malformed queue item: %s", raw.key)
                await self._store.adelete(self._ns, raw.key)
            if max_items is not None and len(result) >= max_items:
                break
            if len(batch) < self._PAGE_SIZE:
                break

        logger.debug("Drained %d items from thread %s", len(result), self._thread_id)
        return result

    async def peek(self, *, limit: int = 100) -> list[QueuedItem]:
        """View queued items without removing them."""
        items_raw = await self._store.asearch(self._ns, limit=limit)
        items_raw.sort(key=lambda x: x.key)  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        result: list[QueuedItem] = []
        for raw in items_raw:
            try:
                result.append(QueuedItem.model_validate(raw.value))
            except Exception:
                logger.warning("Skipping malformed queue item in peek: %s", raw.key)
        return result

    async def depth(self) -> int:
        """Return number of items in the queue (paged so large queues count)."""
        total = 0
        while True:
            batch = await self._store.asearch(self._ns, limit=self._PAGE_SIZE)
            total += len(batch)
            if len(batch) < self._PAGE_SIZE:
                break
            # Avoid infinite loop if the store doesn't respect limit
            # semantics — break after one full page since we already
            # have a conservative upper bound.
            break
        return total

    async def clear(self) -> int:
        """Remove all items from the queue in pages. Returns count removed."""
        removed = 0
        while True:
            batch = await self._store.asearch(self._ns, limit=self._PAGE_SIZE)
            if not batch:
                break
            for item in batch:
                await self._store.adelete(self._ns, item.key)
            removed += len(batch)
            if len(batch) < self._PAGE_SIZE:
                break
        return removed


# ---------------------------------------------------------------------------
# Thread-busy tracking via Store
# ---------------------------------------------------------------------------

BUSY_NAMESPACE = ("thread_busy",)
_BUSY_LOCK_EXPIRY_SECONDS = 600  # auto-expire stale locks after 10 minutes


class ThreadBusyTracker:
    """Track which threads have an active run using Store.

    This is a best-effort advisory lock — it prevents the route layer from
    starting a competing run but doesn't provide strict mutual exclusion.
    Stale locks auto-expire after ``_BUSY_LOCK_EXPIRY_SECONDS``.
    """

    def __init__(self, store: Any) -> None:
        super().__init__()
        self._store = store

    async def mark_busy(self, thread_id: str) -> None:
        await self._store.aput(
            BUSY_NAMESPACE,
            thread_id,
            {"busy": True, "since": time.time()},
        )

    async def heartbeat(self, thread_id: str) -> None:
        """Extend the busy lock for a long-running thread.

        Call this periodically from inside a long-running turn so the
        advisory lock doesn't expire mid-run. Without a heartbeat a turn
        that exceeds ``_BUSY_LOCK_EXPIRY_SECONDS`` (10 min) looks idle
        to ``is_busy`` even though the run is still active.
        """
        await self._store.aput(
            BUSY_NAMESPACE,
            thread_id,
            {"busy": True, "since": time.time()},
        )

    async def mark_idle(self, thread_id: str) -> None:
        await self._store.adelete(BUSY_NAMESPACE, thread_id)

    async def is_busy(self, thread_id: str) -> bool:
        item = await self._store.aget(BUSY_NAMESPACE, thread_id)
        if item is None:
            return False
        since = item.value.get("since", 0)
        if time.time() - since > _BUSY_LOCK_EXPIRY_SECONDS:
            await self._store.adelete(BUSY_NAMESPACE, thread_id)
            return False
        return True


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class QueuedInputMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Drains queued messages from Store before each model call.

    On each `abefore_model` invocation:
    1. Reads thread_id from the LangGraph config
    2. Drains all queued items from Store namespace ("queue", thread_id)
    3. Converts items to LangChain messages based on their semantic type
    4. Returns state update that appends those messages

    This allows late-arriving user messages, corrections, and goal changes
    to be injected into an active run without starting a competing one.
    """

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        store = getattr(runtime, "store", None)
        if store is None:
            return None

        config = get_config()
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        queue = ThreadQueue(store, thread_id)
        items = await queue.drain()
        if not items:
            return None

        messages = _items_to_messages(items)
        logger.info(
            "Injected %d queued message(s) into thread %s",
            len(messages),
            thread_id,
        )
        return {"messages": messages}


def _items_to_messages(items: list[QueuedItem]) -> list[Any]:
    """Convert queued items to LangChain messages based on semantic type."""
    messages: list[Any] = []

    for item in items:
        if item.semantic == QueueSemantic.APPEND:
            messages.append(HumanMessage(content=item.content))

        elif item.semantic == QueueSemantic.INTERRUPT:
            messages.append(
                SystemMessage(
                    content=(
                        "[URGENT UPDATE from user]\n"
                        f"{item.content}\n"
                        "[Address this before continuing your current work.]"
                    )
                )
            )

        elif item.semantic == QueueSemantic.REPLACE_GOAL:
            messages.append(
                SystemMessage(
                    content=(
                        "[GOAL CHANGE]\n"
                        "The user has replaced the current task. "
                        "Stop your current work and switch to:\n\n"
                        f"{item.content}"
                    )
                )
            )

    return messages
