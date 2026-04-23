"""Cluster A edges — async task tool surface reached through the graph.

``register_async_tools`` wires ``start_async_task``, ``check_async_task``,
``cancel_async_task``, and ``list_async_tasks`` into the active tool
surface for every default build. The unit tests in ``test_r1_features``
exercise the :class:`AsyncTaskManager` directly; these tests verify the
LLM-facing tool signatures and error paths through a real compiled graph.

Deep agents built via ``build_deep_agent`` currently pass
``available_graphs=None`` to ``build_async_task_tools`` (the wiring that
lets the agent actually *launch* sub-graphs happens at a higher level).
That means the default-build behavior these tests capture is:

- ``start_async_task`` with any worker type → "not available" error.
- ``list_async_tasks`` → "No background tasks found" when empty.
- ``check_async_task`` / ``cancel_async_task`` with unknown id → "No task found".
- ``list_async_tasks("bogus")`` → invalid-filter error message.

If a future change actually populates ``available_graphs`` at the builder
level, these tests still pass (they cover the empty-config path), and
new tests should be added for the populated-registry happy path.
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
async def test_start_async_task_without_worker_graphs_returns_recoverable_error(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Default build has no worker graphs — start_async_task must not crash the run.

    The tool returns a structured "Worker type 'X' not available" string
    that the LLM can reason about. This is the contract downstream
    wrappers rely on: unavailability is a recoverable tool result, not
    an exception that kills the graph.
    """
    scripted = scripted_llm(
        [
            tool_call_turn(
                "start_async_task",
                {"description": "dig into logs", "worker_type": "researcher"},
            ),
            answer("noted: no workers available"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="async-noop",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="launch something")]},
        config={"configurable": {"thread_id": "async-noop"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "start_async_task")
    content = str(tool_msg.content).lower()
    assert "not available" in content, (
        f"start_async_task with no worker graphs should return the"
        f" 'not available' string; got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_list_async_tasks_empty_returns_structured_message(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """list_async_tasks on a fresh thread returns the 'none found' string."""
    scripted = scripted_llm(
        [
            tool_call_turn("list_async_tasks"),
            answer("none tracked"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="async-list-empty",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="any running?")]},
        config={"configurable": {"thread_id": "async-list-empty"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "list_async_tasks")
    content = str(tool_msg.content).lower()
    assert "no background tasks" in content, (
        f"Empty list_async_tasks should surface the kit's standard"
        f" empty-state string; got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_check_async_task_with_unknown_id_returns_not_found(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """check_async_task with a bogus id doesn't crash — returns a recoverable error."""
    scripted = scripted_llm(
        [
            tool_call_turn("check_async_task", {"task_id": "does-not-exist"}),
            answer("gone"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="async-check-unknown",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="check it")]},
        config={"configurable": {"thread_id": "async-check-unknown"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "check_async_task")
    content = str(tool_msg.content).lower()
    assert "no task found" in content, (
        f"check_async_task with unknown id should surface the 'No task"
        f" found' sentinel; got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_list_async_tasks_invalid_status_filter_is_recoverable(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """An invalid status filter returns a structured error the LLM can reason about."""
    scripted = scripted_llm(
        [
            tool_call_turn("list_async_tasks", {"status_filter": "totally-bogus"}),
            answer("retrying"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="async-bad-filter",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="list with bogus filter")]},
        config={"configurable": {"thread_id": "async-bad-filter"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "list_async_tasks")
    content = str(tool_msg.content).lower()
    assert "invalid" in content, (
        f"Bogus status_filter should surface the kit's 'Invalid status"
        f" filter' error; got {tool_msg.content!r}"
    )


@pytest.mark.asyncio
async def test_cancel_async_task_with_unknown_id_returns_not_found(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """cancel_async_task with an unknown task id returns the sentinel, not an exception."""
    scripted = scripted_llm(
        [
            tool_call_turn("cancel_async_task", {"task_id": "nope"}),
            answer("ok"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="async-cancel-unknown",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="cancel it")]},
        config={"configurable": {"thread_id": "async-cancel-unknown"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "cancel_async_task")
    content = str(tool_msg.content).lower()
    assert "no task found" in content, (
        f"cancel_async_task with unknown id should surface 'No task"
        f" found'; got {tool_msg.content!r}"
    )
