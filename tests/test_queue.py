"""Tests for busy-thread queueing middleware and Store-backed queue."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_kit.core.orchestration.queue import (
    QueuedInputMiddleware,
    QueuedItem,
    QueueSemantic,
    ThreadBusyTracker,
    ThreadQueue,
    _items_to_messages,
)

# ---------------------------------------------------------------------------
# ThreadQueue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_and_drain(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-1")

    await queue.enqueue(QueuedItem(content="Hello"))
    await queue.enqueue(QueuedItem(content="World"))

    assert await queue.depth() == 2

    items = await queue.drain()
    assert len(items) == 2
    assert items[0].content == "Hello"
    assert items[1].content == "World"

    # Queue should be empty after drain
    assert await queue.depth() == 0


@pytest.mark.asyncio
async def test_drain_empty_queue(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-empty")
    items = await queue.drain()
    assert items == []


@pytest.mark.asyncio
async def test_peek_does_not_remove(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-2")
    await queue.enqueue(QueuedItem(content="Peeked"))

    peeked = await queue.peek()
    assert len(peeked) == 1
    assert peeked[0].content == "Peeked"

    # Still there
    assert await queue.depth() == 1


@pytest.mark.asyncio
async def test_clear(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-3")
    await queue.enqueue(QueuedItem(content="One"))
    await queue.enqueue(QueuedItem(content="Two"))

    removed = await queue.clear()
    assert removed == 2
    assert await queue.depth() == 0


@pytest.mark.asyncio
async def test_replace_goal_clears_queue_first(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-4")
    await queue.enqueue(QueuedItem(content="Old task 1"))
    await queue.enqueue(QueuedItem(content="Old task 2"))

    await queue.enqueue(
        QueuedItem(content="New goal", semantic=QueueSemantic.REPLACE_GOAL)
    )

    items = await queue.drain()
    # Only the replace_goal item should remain
    assert len(items) == 1
    assert items[0].content == "New goal"
    assert items[0].semantic == QueueSemantic.REPLACE_GOAL


@pytest.mark.asyncio
async def test_fifo_ordering(mock_store: Any) -> None:
    queue = ThreadQueue(mock_store, "thread-5")

    # Enqueue with explicit timestamps to test ordering
    item1 = QueuedItem(content="First", timestamp=1000.0)
    item2 = QueuedItem(content="Second", timestamp=1001.0)
    item3 = QueuedItem(content="Third", timestamp=1002.0)

    await queue.enqueue(item2)
    await queue.enqueue(item1)
    await queue.enqueue(item3)

    items = await queue.drain()
    assert [i.content for i in items] == ["First", "Second", "Third"]


@pytest.mark.asyncio
async def test_queues_are_isolated_per_thread(mock_store: Any) -> None:
    q1 = ThreadQueue(mock_store, "thread-a")
    q2 = ThreadQueue(mock_store, "thread-b")

    await q1.enqueue(QueuedItem(content="For A"))
    await q2.enqueue(QueuedItem(content="For B"))

    items_a = await q1.drain()
    items_b = await q2.drain()

    assert len(items_a) == 1
    assert items_a[0].content == "For A"
    assert len(items_b) == 1
    assert items_b[0].content == "For B"


# ---------------------------------------------------------------------------
# ThreadBusyTracker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_busy_and_idle(mock_store: Any) -> None:
    tracker = ThreadBusyTracker(mock_store)

    assert not await tracker.is_busy("thread-1")

    await tracker.mark_busy("thread-1")
    assert await tracker.is_busy("thread-1")

    await tracker.mark_idle("thread-1")
    assert not await tracker.is_busy("thread-1")


@pytest.mark.asyncio
async def test_stale_lock_auto_expires(mock_store: Any) -> None:
    tracker = ThreadBusyTracker(mock_store)

    # Manually write a stale lock (11 minutes ago)
    import time

    await mock_store.aput(
        ("thread_busy",),
        "thread-stale",
        {"busy": True, "since": time.time() - 700},
    )

    # Should auto-expire
    assert not await tracker.is_busy("thread-stale")


# ---------------------------------------------------------------------------
# _items_to_messages conversion
# ---------------------------------------------------------------------------


def test_append_converts_to_human_message() -> None:
    items = [QueuedItem(content="Hello there", semantic=QueueSemantic.APPEND)]
    messages = _items_to_messages(items)

    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello there"


def test_interrupt_converts_to_system_message() -> None:
    items = [QueuedItem(content="Stop!", semantic=QueueSemantic.INTERRUPT)]
    messages = _items_to_messages(items)

    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "URGENT UPDATE" in messages[0].content
    assert "Stop!" in messages[0].content


def test_replace_goal_converts_to_system_message() -> None:
    items = [QueuedItem(content="Do X instead", semantic=QueueSemantic.REPLACE_GOAL)]
    messages = _items_to_messages(items)

    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "GOAL CHANGE" in messages[0].content
    assert "Do X instead" in messages[0].content


def test_mixed_semantics() -> None:
    items = [
        QueuedItem(content="Follow up", semantic=QueueSemantic.APPEND),
        QueuedItem(content="Urgent fix", semantic=QueueSemantic.INTERRUPT),
        QueuedItem(content="New direction", semantic=QueueSemantic.REPLACE_GOAL),
    ]
    messages = _items_to_messages(items)

    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], SystemMessage)
    assert isinstance(messages[2], SystemMessage)


# ---------------------------------------------------------------------------
# QueuedInputMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_drains_queue(mock_store: Any, monkeypatch: Any) -> None:
    """Test that the middleware drains queued items and returns messages."""
    # Pre-populate queue
    queue = ThreadQueue(mock_store, "test-thread")
    await queue.enqueue(QueuedItem(content="Queued message"))

    middleware = QueuedInputMiddleware()

    # Mock runtime with store
    class FakeRuntime:
        store = mock_store

    # Mock get_config to return our thread_id
    monkeypatch.setattr(
        "langgraph_kit.core.orchestration.queue.get_config",
        lambda: {"configurable": {"thread_id": "test-thread"}},
    )

    result = await middleware.abefore_model({}, FakeRuntime())

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Queued message"

    # Queue should be drained
    assert await queue.depth() == 0


@pytest.mark.asyncio
async def test_middleware_returns_none_when_empty(
    mock_store: Any, monkeypatch: Any
) -> None:
    middleware = QueuedInputMiddleware()

    class FakeRuntime:
        store = mock_store

    monkeypatch.setattr(
        "langgraph_kit.core.orchestration.queue.get_config",
        lambda: {"configurable": {"thread_id": "empty-thread"}},
    )

    result = await middleware.abefore_model({}, FakeRuntime())
    assert result is None


@pytest.mark.asyncio
async def test_middleware_returns_none_without_store() -> None:
    middleware = QueuedInputMiddleware()

    class NoStoreRuntime:
        store = None

    result = await middleware.abefore_model({}, NoStoreRuntime())
    assert result is None


@pytest.mark.asyncio
async def test_middleware_returns_none_without_thread_id(
    mock_store: Any, monkeypatch: Any
) -> None:
    middleware = QueuedInputMiddleware()

    class FakeRuntime:
        store = mock_store

    monkeypatch.setattr(
        "langgraph_kit.core.orchestration.queue.get_config",
        lambda: {"configurable": {}},
    )

    result = await middleware.abefore_model({}, FakeRuntime())
    assert result is None
