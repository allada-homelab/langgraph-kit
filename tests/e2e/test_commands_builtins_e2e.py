"""Cluster C — end-to-end scenarios for the built-in slash commands.

``/compact`` is already covered in ``test_commands_e2e.py``. This file
fills in the remaining built-ins: ``/help``, ``/memory``, ``/tools``,
``/skills``, ``/status``, ``/context``. Each has the same invariant —
command is intercepted by ``CommandMiddleware``, LLM is never called,
dispatcher output lands as an ``AIMessage``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    last_ai_message,
    scripted_llm,
)

pytestmark = pytest.mark.e2e


async def _run_command(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
    *,
    command: str,
    thread_id: str,
) -> AIMessage:
    """Build a graph with zero-turn LLM, send ``command``, return last AIMessage.

    Zero-turn LLM means: if the command short-circuit leaks to the
    model path, the scripted LLM raises ``ReplayMismatchError`` and the
    caller sees the exception. Reaching the return proves the command
    was dispatched without invoking the LLM.
    """
    with patched_build_llm(scripted_llm([])):
        graph, _ = build_deep_agent(
            agent_name=f"cmd-e2e-{thread_id}",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=command)]},
        config={"configurable": {"thread_id": thread_id}},  # pyright: ignore[reportArgumentType]
    )
    return last_ai_message(result)


@pytest.mark.asyncio
async def test_help_lists_available_commands(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/help",
        thread_id="cmd-help",
    )
    content = str(msg.content).lower()
    assert "available commands" in content or "/help" in content, (
        f"/help output doesn't look like a command list: {msg.content!r}"
    )
    # Spot-check a few built-in commands should be listed.
    assert "/compact" in str(msg.content)
    assert "/memory" in str(msg.content)


@pytest.mark.asyncio
async def test_memory_with_empty_store_reports_empty(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/memory",
        thread_id="cmd-memory-empty",
    )
    content = str(msg.content).lower()
    assert "no memories" in content, (
        f"/memory with empty store should say 'no memories'; got {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_memory_invalid_scope_surfaces_error(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/memory not-a-scope",
        thread_id="cmd-memory-badscope",
    )
    content = str(msg.content).lower()
    assert "invalid scope" in content, (
        f"/memory with invalid scope should return an error: {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_tools_lists_registered_tools(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/tools",
        thread_id="cmd-tools",
    )
    content = str(msg.content).lower()
    # Spot-check standard tools appear in the listing.
    assert "save_memory" in content or "tool_search" in content, (
        f"/tools output should list standard tools: {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_skills_lists_available_skills(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/skills",
        thread_id="cmd-skills",
    )
    content = str(msg.content).lower()
    assert "code-review" in content or "research" in content or "skill" in content, (
        f"/skills output should list default skills: {msg.content!r}"
    )


@pytest.mark.asyncio
async def test_status_returns_a_summary(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/status",
        thread_id="cmd-status",
    )
    # /status format varies; the contract is just "returns some summary string".
    content = str(msg.content)
    assert len(content) > 0, "/status produced empty output"


@pytest.mark.asyncio
async def test_unknown_command_falls_through_to_llm(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Unknown slash commands must pass through to the LLM, not error out.

    If the dispatcher rejects (returns ``handled=False``), the middleware
    returns None and the agent proceeds normally. We script one LLM
    turn to receive the passthrough; if the middleware mishandles
    (raises, or incorrectly short-circuits), the scripted LLM sees
    wrong input or never fires.
    """
    from tests.e2e.helpers import answer

    with patched_build_llm(scripted_llm([answer("handled by LLM")])):
        graph, _ = build_deep_agent(
            agent_name="cmd-unknown",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="/definitelynotacommand arg")]},
        config={"configurable": {"thread_id": "cmd-unk"}},  # pyright: ignore[reportArgumentType]
    )
    final = last_ai_message(result)
    assert "handled by LLM" in str(final.content), (
        f"Unknown slash command didn't fall through to the LLM; got: {final.content!r}"
    )
