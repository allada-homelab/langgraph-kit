"""Cluster A — end-to-end memory tool flows.

Exercises the full save / list / update / delete round-trip through a
running graph. The kit has unit-level coverage for each operation; the
e2e layer here catches composition bugs: correct wiring from scripted
LLM → tool → PersistentMemoryManager → Store → next-turn tool that
reads back what was written.
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


def _extract_id(tool_message_content: Any) -> str:
    """Pull the memory id out of a save/update ToolMessage.

    Tool bodies format their confirmation as
    ``"Memory saved: [type] title (id: <id>)"``. This helper parses that
    shape out for follow-up tool calls in the same scripted turn.
    """
    text = str(tool_message_content)
    marker = "(id: "
    idx = text.find(marker)
    if idx == -1:
        msg = f"Could not extract memory id from: {text!r}"
        raise AssertionError(msg)
    start = idx + len(marker)
    end = text.find(")", start)
    return text[start:end]


@pytest.mark.asyncio
async def test_save_then_list_memories(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """LLM saves two memories then lists them — both appear in the list result."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "save_memory",
                {
                    "title": "pi",
                    "memory_type": "reference",
                    "scope": "user",
                    "summary": "approximate value of pi",
                    "body": "pi=3.14159",
                },
                call_id="c1",
            ),
            tool_call_turn(
                "save_memory",
                {
                    "title": "e",
                    "memory_type": "reference",
                    "scope": "user",
                    "summary": "euler's number",
                    "body": "e=2.71828",
                },
                call_id="c2",
            ),
            tool_call_turn(
                "list_memories",
                {"scope": "user"},
                call_id="c3",
            ),
            answer("listed"),
        ]
    )

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="mem-crud-list",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="remember two things")]},
        config={"configurable": {"thread_id": "mem-1"}},  # pyright: ignore[reportArgumentType]
    )

    assert_tool_invoked(result, "save_memory")

    # list_memories ToolMessage should reference both saved titles.
    list_msg = assert_tool_invoked(result, "list_memories")
    content = str(list_msg.content)
    assert "pi" in content, f"'pi' record missing from list: {content!r}"
    assert "e" in content, f"'e' record missing from list: {content!r}"
    assert "Found 2 memories" in content or "2 memor" in content, (
        f"list_memories count summary missing: {content!r}"
    )


@pytest.mark.asyncio
async def test_save_update_then_read_shows_update(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """After an update, a subsequent list/search sees the new body — not the old one.

    Verifies update actually mutates the stored record rather than
    creating a duplicate or silently failing. Since scripted LLM turns
    are sequential and `update_memory` needs the record's id (from
    save's response), we embed a known id in the call by reusing
    deterministic ids via a two-pass approach: save first (to discover
    the assigned id by inspecting the ToolMessage), then build a second
    graph with a second scripted LLM that does the update.

    The simpler pattern — use `list_memories` between save and update
    to discover the id — doesn't work inside a single scripted script
    because the script is fixed upfront. Two invocations on the same
    thread / store is the cleanest path.
    """
    # Pass 1: save the memory.
    scripted_save = scripted_llm(
        [
            tool_call_turn(
                "save_memory",
                {
                    "title": "approx",
                    "memory_type": "reference",
                    "scope": "user",
                    "summary": "initial value",
                    "body": "v=1",
                },
            ),
            answer("saved"),
        ]
    )
    with patched_build_llm(scripted_save):
        graph, _ = build_deep_agent(
            agent_name="mem-crud-update",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    pass1 = await graph.ainvoke(
        {"messages": [HumanMessage(content="save")]},
        config={"configurable": {"thread_id": "mem-update"}},  # pyright: ignore[reportArgumentType]
    )
    save_msg = assert_tool_invoked(pass1, "save_memory")
    mem_id = _extract_id(save_msg.content)

    # Pass 2: update the just-saved record by id.
    scripted_update = scripted_llm(
        [
            tool_call_turn(
                "update_memory",
                {
                    "memory_id": mem_id,
                    "scope": "user",
                    "body": "v=2",
                    "summary": "updated value",
                },
            ),
            tool_call_turn("list_memories", {"scope": "user"}),
            answer("updated"),
        ]
    )
    with patched_build_llm(scripted_update):
        graph2, _ = build_deep_agent(
            agent_name="mem-crud-update",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    pass2 = await graph2.ainvoke(
        {"messages": [HumanMessage(content="update")]},
        config={"configurable": {"thread_id": "mem-update"}},  # pyright: ignore[reportArgumentType]
    )

    update_msg = assert_tool_invoked(pass2, "update_memory")
    assert "updated" in str(update_msg.content).lower() or mem_id in str(
        update_msg.content
    ), f"update_memory didn't confirm the update: {update_msg.content!r}"

    # list_memories result should mention the new summary, not the old
    # one — evidence the update actually mutated the stored record.
    list_msg = assert_tool_invoked(pass2, "list_memories")
    assert "updated value" in str(list_msg.content), (
        f"list_memories shows stale summary after update: {list_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_delete_memory_removes_it_from_list(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Full save → delete → list round-trip: deleted record disappears from list."""
    # Pass 1: save.
    with patched_build_llm(
        scripted_llm(
            [
                tool_call_turn(
                    "save_memory",
                    {
                        "title": "to-delete",
                        "memory_type": "reference",
                        "scope": "user",
                        "summary": "ephemeral",
                        "body": "x=1",
                    },
                ),
                answer("saved"),
            ]
        )
    ):
        graph_save, _ = build_deep_agent(
            agent_name="mem-crud-del",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    saved_state = await graph_save.ainvoke(
        {"messages": [HumanMessage(content="save")]},
        config={"configurable": {"thread_id": "mem-del"}},  # pyright: ignore[reportArgumentType]
    )
    mem_id = _extract_id(assert_tool_invoked(saved_state, "save_memory").content)

    # Pass 2: delete + list.
    with patched_build_llm(
        scripted_llm(
            [
                tool_call_turn(
                    "delete_memory",
                    {"memory_id": mem_id, "scope": "user"},
                ),
                tool_call_turn("list_memories", {"scope": "user"}),
                answer("deleted"),
            ]
        )
    ):
        graph_del, _ = build_deep_agent(
            agent_name="mem-crud-del",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    deleted_state = await graph_del.ainvoke(
        {"messages": [HumanMessage(content="delete")]},
        config={"configurable": {"thread_id": "mem-del"}},  # pyright: ignore[reportArgumentType]
    )

    delete_msg = assert_tool_invoked(deleted_state, "delete_memory")
    assert "deleted" in str(delete_msg.content).lower(), (
        f"delete_memory didn't confirm: {delete_msg.content!r}"
    )

    list_msg = assert_tool_invoked(deleted_state, "list_memories")
    content = str(list_msg.content)
    assert "to-delete" not in content, (
        f"Deleted record still appears in list: {content!r}"
    )
    assert (
        "No memories found" in content
        or "Found 0" in content
        or "to-delete" not in content
    )


@pytest.mark.asyncio
async def test_save_memory_with_invalid_type_returns_error_to_llm(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Invalid memory_type must be surfaced as an error string, not a crash.

    The tool returns ``Error: invalid memory_type 'xyz'``; the LLM can
    observe the message and try again. If a future refactor raised
    ValueError instead, the whole run would crash and the agent could
    not recover.
    """
    scripted = scripted_llm(
        [
            tool_call_turn(
                "save_memory",
                {
                    "title": "bad",
                    "memory_type": "not-a-real-type",
                    "scope": "user",
                    "summary": "x",
                    "body": "x",
                },
            ),
            answer("handled"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="mem-bad-type",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="save")]},
        config={"configurable": {"thread_id": "mem-bad"}},  # pyright: ignore[reportArgumentType]
    )

    msg = assert_tool_invoked(result, "save_memory")
    content = str(msg.content).lower()
    assert "error" in content, (
        f"Expected 'error' in save_memory response: {msg.content!r}"
    )
    assert "memory_type" in content, (
        f"Expected 'memory_type' in error message: {msg.content!r}"
    )
