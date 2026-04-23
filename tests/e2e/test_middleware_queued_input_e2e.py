"""Cluster B edges — ``QueuedInputMiddleware`` drains store queue into the turn.

The queue endpoints (``POST /agents/{id}/threads/{tid}/queue``) enqueue
``HumanMessage`` / ``SystemMessage`` payloads into the store under
namespace ``("queue", thread_id)``. ``QueuedInputMiddleware.abefore_model``
drains that namespace at the start of every model call and returns
``{"messages": [...]}`` so the new messages land in state before the LLM
sees them.

These tests use a ``CapturingScriptedChatModel`` so we can inspect the
exact input the LLM received — the whole point of the middleware is that
the queued content SHOULD have made it into the prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.orchestration.queue import (
    QueuedItem,
    QueueSemantic,
    ThreadQueue,
)
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    capturing_scripted_llm,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_append_queued_item_reaches_the_llm_as_human_message(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """APPEND semantic → queued content reaches the LLM on its next call."""
    thread_id = "queued-append"
    queue = ThreadQueue(e2e_store, thread_id)
    marker = "QUEUED-APPEND-MARKER-9f3b"
    await queue.enqueue(
        QueuedItem(content=f"reminder: {marker}", semantic=QueueSemantic.APPEND)
    )

    capturing = capturing_scripted_llm([answer("ack")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="queued-append-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="main ask")]},
        config={"configurable": {"thread_id": thread_id}},  # pyright: ignore[reportArgumentType]
    )

    assert capturing.captured_calls, "scripted model was never invoked"
    seen = "\n".join(
        str(getattr(m, "content", "")) for m in capturing.captured_calls[0]
    )
    assert marker in seen, (
        f"Queued APPEND content should have reached the LLM prompt;"
        f" marker {marker!r} not in captured messages. Excerpt: {seen[:500]!r}"
    )

    # The queue must also be drained — a redelivery would re-inject the
    # same content on every subsequent turn.
    depth = await queue.depth()
    assert depth == 0, f"Queue should be drained after injection; depth={depth}"


@pytest.mark.asyncio
async def test_interrupt_queued_item_reaches_llm_as_urgent_system_message(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """INTERRUPT semantic → queued content delivered with URGENT framing."""
    thread_id = "queued-interrupt"
    queue = ThreadQueue(e2e_store, thread_id)
    marker = "URGENT-MARKER-a7d1"
    await queue.enqueue(
        QueuedItem(
            content=f"fix this first: {marker}", semantic=QueueSemantic.INTERRUPT
        )
    )

    capturing = capturing_scripted_llm([answer("on it")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="queued-interrupt-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="do the original thing")]},
        config={"configurable": {"thread_id": thread_id}},  # pyright: ignore[reportArgumentType]
    )

    seen = "\n".join(
        str(getattr(m, "content", "")) for m in capturing.captured_calls[0]
    )
    assert "URGENT UPDATE" in seen, (
        "INTERRUPT semantic should produce the [URGENT UPDATE from user] framing"
    )
    assert marker in seen, (
        f"Interrupt body not delivered; expected {marker!r} in captured messages"
    )


@pytest.mark.asyncio
async def test_empty_queue_is_no_op(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """No queued items → middleware returns None, graph runs normally."""
    capturing = capturing_scripted_llm([answer("nothing to do")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="queued-empty-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "queued-empty"}},  # pyright: ignore[reportArgumentType]
    )

    assert "messages" in result, "graph should have completed normally"
    # No queue-side effects should have fired.
    assert capturing.captured_calls, "scripted model was never invoked"
