"""Cluster A edges — ``retrieve_result`` round-trip through a real graph.

``ResultPersistenceMiddleware`` offloads large tool outputs to
``("tool_results", thread_id)`` and replaces the inline ToolMessage with a
preview + reference key. ``retrieve_result(result_ref=KEY)`` is how the LLM
pulls the full content back on demand. Until now the persist side was
covered (``test_middleware_stack_e2e.py``) but the retrieve side was not.

The scripted LLM can't know the runtime-generated ref key in advance, so
this test pre-populates the per-thread namespace directly on the store and
scripts the LLM to call ``retrieve_result(result_ref="known_key")``. That
isolates the retrieval side of the contract from the hashing side.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_retrieve_result_returns_full_persisted_content(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """LLM calls retrieve_result with a known ref → ToolMessage carries the full content."""
    # Pre-populate the persistence namespace (per-thread) with a known record.
    full_content = "LONG-CONTENT-MARKER " + ("x" * 8000)
    await e2e_store.aput(
        ("tool_results", "retrieve-1"),
        "known_key",
        {
            "content": full_content,
            "tool_name": "big_upstream",
            "tool_call_id": "synthetic",
            "char_count": len(full_content),
        },
    )

    scripted = scripted_llm(
        [
            tool_call_turn(
                "retrieve_result",
                {"result_ref": "known_key"},
            ),
            answer("got it"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="retrieve-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="fetch it")]},
        config={"configurable": {"thread_id": "retrieve-1"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "retrieve_result")
    content = str(tool_msg.content)
    assert "LONG-CONTENT-MARKER" in content, (
        f"retrieve_result should have returned the full persisted content;"
        f" got {content[:200]!r}"
    )
    assert "big_upstream" in content, "Retrieval header should identify the source tool"


@pytest.mark.asyncio
async def test_retrieve_result_unknown_key_returns_not_found(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Unknown result_ref → recoverable 'not found' message, not an exception."""
    scripted = scripted_llm(
        [
            tool_call_turn("retrieve_result", {"result_ref": "ghost-ref"}),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="retrieve-missing",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="fetch ghost")]},
        config={"configurable": {"thread_id": "retrieve-missing"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "retrieve_result")
    content = str(tool_msg.content).lower()
    assert "no persisted result" in content, (
        f"Missing-key retrieve should surface the 'No persisted result'"
        f" sentinel; got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_retrieve_result_honors_offset_and_limit(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """offset + limit paginate through the stored content."""
    # Distinguishable pattern: first 1000 chars are A's, next 1000 are B's,
    # then C's. Offset=1000 limit=1000 should return only B's plus header.
    full_content = ("A" * 1000) + ("B" * 1000) + ("C" * 1000)
    await e2e_store.aput(
        ("tool_results", "retrieve-paged"),
        "paged",
        {
            "content": full_content,
            "tool_name": "pages",
            "tool_call_id": "synthetic",
            "char_count": len(full_content),
        },
    )

    scripted = scripted_llm(
        [
            tool_call_turn(
                "retrieve_result",
                {"result_ref": "paged", "offset": 1000, "limit": 1000},
            ),
            answer("page 2"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="retrieve-paged",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="page 2 please")]},
        config={"configurable": {"thread_id": "retrieve-paged"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "retrieve_result")
    content = str(tool_msg.content)
    assert "B" * 1000 in content, (
        "Page 2 should contain the B-block (offset=1000, limit=1000)"
    )
    # The A-block appears in the header as "chars 1000-..." but not as a
    # thousand-A substring in the body. Make sure the chunk boundary holds.
    assert "A" * 500 not in content, (
        "Page 2 body should not include the A-block (offset boundary violated)"
    )
    assert "C" * 500 not in content, (
        "Page 2 body should not include the C-block (limit boundary violated)"
    )
    assert "remaining" in content.lower(), (
        "Paginated output should advertise remaining chars"
    )
