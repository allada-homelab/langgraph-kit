"""Cluster A — end-to-end UI tool scenarios.

``create_artifact``, ``emit_progress``, ``suggest_actions``, and
``add_citation`` are all registered in the standard tool bundle. They
don't mutate state — they emit sentinel-prefixed strings that the
streaming layer picks up — so the e2e-level check is "the tool runs,
returns a sentinel, and doesn't crash under realistic inputs."
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
async def test_create_artifact_round_trip(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """create_artifact returns a sentinel-prefixed string for the streaming layer."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "create_artifact",
                {
                    "artifact_type": "code",
                    "title": "hello.py",
                    "content": "print('hi')",
                    "language": "python",
                },
            ),
            answer("done"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="ui-artifact",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="show code")]},
        config={"configurable": {"thread_id": "ui-artifact"}},  # pyright: ignore[reportArgumentType]
    )
    msg = assert_tool_invoked(result, "create_artifact")
    content = str(msg.content)
    assert "hello.py" in content or "print" in content or "artifact" in content.lower(), (
        f"create_artifact produced unexpected content: {content!r}"
    )


@pytest.mark.asyncio
async def test_emit_progress_rejects_invalid_counters(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Invalid progress counters return structured error strings, not crashes.

    ``current > total``, ``total < 1``, and empty step all become
    user-visible error strings that the LLM can recover from. The e2e
    layer verifies that error path is preserved end-to-end rather than
    degraded to an exception somewhere in the stack.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("emit_progress", {"step": "x", "current": 5, "total": 2}),
            answer("recovered"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="ui-progress-bad",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="progress")]},
        config={"configurable": {"thread_id": "ui-progress"}},  # pyright: ignore[reportArgumentType]
    )
    msg = assert_tool_invoked(result, "emit_progress")
    content = str(msg.content).lower()
    assert "error" in content, (
        f"Expected 'error' in emit_progress response: {msg.content!r}"
    )
    assert "current" in content, (
        f"Expected error to reference 'current': {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_suggest_actions_round_trip(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """suggest_actions accepts a list of labels and returns a sentinel payload."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "suggest_actions",
                {"actions": ["Run tests", "Review changes"]},
            ),
            answer("suggested"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="ui-sugg",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="suggest")]},
        config={"configurable": {"thread_id": "ui-sugg"}},  # pyright: ignore[reportArgumentType]
    )
    msg = assert_tool_invoked(result, "suggest_actions")
    assert "Run tests" in str(msg.content) or "suggest" in str(msg.content).lower(), (
        f"suggest_actions content unexpected: {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_add_citation_round_trip(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """add_citation formats a reference that the streaming layer renders as a source card."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "add_citation",
                {
                    "title": "auth.py:42",
                    "source": "/repo/auth.py",
                    "snippet": "def authenticate(token): ...",
                },
            ),
            answer("cited"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="ui-cite",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="cite")]},
        config={"configurable": {"thread_id": "ui-cite"}},  # pyright: ignore[reportArgumentType]
    )
    msg = assert_tool_invoked(result, "add_citation")
    assert "auth.py" in str(msg.content) or "cite" in str(msg.content).lower(), (
        f"add_citation content unexpected: {msg.content!r}"
    )
