"""Cluster A edges — HITL ``approve_action`` pause / resume round-trip.

``approve_action`` calls LangGraph's ``interrupt()`` to pause the graph
mid-run. The frontend renders an approval banner and the ``/resume``
endpoint feeds a ``Command(resume=...)`` back in. These tests drive the
full loop with a scripted LLM and in-memory checkpointer so the pause /
resume contract has a regression guard.

What each payload shape the tool is built to accept ends up as:
- ``{"type": "accept"}`` → "User accepted the action."
- ``{"type": "response", "args": "why?"}`` → "User rejected the action with message: why?"
- ``{"type": "ignore"}`` → "User ignored the action. Do not proceed."
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command  # pyright: ignore[reportMissingImports]

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    last_ai_message,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


def _approve_tool_call() -> dict[str, Any]:
    return tool_call_turn(
        "approve_action",
        {
            "action": "delete_record",
            "description": "Delete the legacy record.",
            "action_args": {"id": "rec-123"},
        },
    )


async def _run_with_interrupt_and_resume(
    graph: Any,
    thread_id: str,
    resume_payload: dict[str, Any],
) -> Any:
    """Invoke until the graph pauses at approve_action, then resume once."""
    config = {"configurable": {"thread_id": thread_id}}
    # The first invoke pauses at the interrupt() call inside approve_action.
    # LangGraph's ainvoke returns the paused state (it does NOT raise).
    await graph.ainvoke(
        {"messages": [HumanMessage(content="handle it")]},
        config=config,  # pyright: ignore[reportArgumentType]
    )
    # Resume with the caller's payload shape — the tool's _format_response
    # turns it into the string the LLM sees on the next turn.
    return await graph.ainvoke(
        Command(resume=resume_payload),
        config=config,  # pyright: ignore[reportArgumentType]
    )


@pytest.mark.asyncio
async def test_approve_action_accept_path(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """User accepts → tool returns the 'User accepted' string, run continues."""
    scripted = scripted_llm(
        [
            _approve_tool_call(),
            answer("approval confirmed; done"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="hitl-accept",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await _run_with_interrupt_and_resume(
        graph,
        "hitl-accept",
        {"type": "accept"},
    )

    approval_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "approve_action"
    ]
    assert approval_msgs, (
        "approve_action should have produced a ToolMessage after resume"
    )
    assert "accepted" in str(approval_msgs[-1].content).lower(), (
        f"Accept resume should produce the 'User accepted' string;"
        f" got {approval_msgs[-1].content!r}"
    )
    assert "done" in str(last_ai_message(result).content).lower(), (
        "Agent should have continued after the approval resumed it"
    )


@pytest.mark.asyncio
async def test_approve_action_response_path_carries_user_reason(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """``{"type": "response", "args": "nope"}`` surfaces the reason to the agent."""
    scripted = scripted_llm(
        [
            _approve_tool_call(),
            answer("aborted per user"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="hitl-response",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await _run_with_interrupt_and_resume(
        graph,
        "hitl-response",
        {"type": "response", "args": "too risky right now"},
    )

    approval_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "approve_action"
    ]
    assert approval_msgs, "approve_action should have produced a ToolMessage"
    content = str(approval_msgs[-1].content)
    assert "rejected" in content.lower(), f"Response should carry 'rejected': {content!r}"
    assert "too risky right now" in content, (
        f"Response should include the user's message verbatim: {content!r}"
    )


@pytest.mark.asyncio
async def test_approve_action_ignore_path(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """``{"type": "ignore"}`` returns the 'Do not proceed' instruction."""
    scripted = scripted_llm(
        [
            _approve_tool_call(),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="hitl-ignore",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await _run_with_interrupt_and_resume(
        graph,
        "hitl-ignore",
        {"type": "ignore"},
    )

    approval_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "approve_action"
    ]
    assert approval_msgs
    assert "ignored" in str(approval_msgs[-1].content).lower(), (
        f"Ignore resume should produce the 'User ignored' string;"
        f" got {approval_msgs[-1].content!r}"
    )
