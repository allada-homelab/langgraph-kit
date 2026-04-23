"""Cluster A — ``search_memories`` tool e2e round-trip.

The only standard memory tool without explicit e2e coverage. Unit tests
at ``test_memory.py`` cover ``PersistentMemoryManager.search`` directly;
this file verifies the tool wraps that manager and surfaces results /
errors in the shape the LLM sees.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.memory.models import MemoryRecord, MemoryScope, MemoryType
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_search_memories_returns_matching_records(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Populated store → ``search_memories`` returns the persisted content."""
    mgr = PersistentMemoryManager(e2e_store)
    await mgr.create(
        MemoryRecord(
            title="Tacos preference",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
            summary="User likes tacos",
            body="The user has expressed they really enjoy tacos.",
        )
    )

    scripted = scripted_llm(
        [
            tool_call_turn("search_memories", {"query": "food"}),
            answer("found it"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="search-mem-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="what do you know about food?")]},
        config={"configurable": {"thread_id": "search-mem"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "search_memories")
    content = str(tool_msg.content)
    assert "Tacos preference" in content, (
        f"search_memories should carry the record title back to the LLM;"
        f" got {content!r}"
    )
    assert "tacos" in content.lower(), "Body/summary should appear in the search result"


@pytest.mark.asyncio
async def test_search_memories_on_empty_store_returns_sentinel(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Empty store → the 'No matching memories found.' sentinel."""
    scripted = scripted_llm(
        [
            tool_call_turn("search_memories", {"query": "anything"}),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="search-mem-empty",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="any memories?")]},
        config={"configurable": {"thread_id": "search-mem-empty"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "search_memories")
    assert "no matching memories" in str(tool_msg.content).lower(), (
        f"Empty search should surface the 'No matching memories' sentinel;"
        f" got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_search_memories_with_invalid_scope_is_recoverable(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """An invalid ``scope`` returns a structured error message, not an exception."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "search_memories",
                {"query": "anything", "scope": "not-a-scope"},
            ),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="search-mem-badscope",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="search with bad scope")]},
        config={"configurable": {"thread_id": "search-mem-badscope"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "search_memories")
    content = str(tool_msg.content).lower()
    assert "invalid scope" in content, (
        f"Bogus scope should surface the 'invalid scope' sentinel; got {tool_msg.content!r}"
    )
