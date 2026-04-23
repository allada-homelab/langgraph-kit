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
async def test_tools_command_filters_by_tag(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """``/tools <tag>`` narrows the listing to tools matching the tag."""
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/tools memory",
        thread_id="cmd-tools-tag",
    )
    content = str(msg.content)
    # Every memory tool carries the "memory" tag; the listing should be
    # non-empty and not mention tools from other groups.
    assert "save_memory" in content
    assert "tool_search" not in content


@pytest.mark.asyncio
async def test_tools_command_with_unknown_tag_returns_empty_listing(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A tag nothing matches returns a 0-count header, not a crash."""
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/tools nonexistent-tag-8f3a",
        thread_id="cmd-tools-notag",
    )
    content = str(msg.content)
    assert "Registered Tools (0)" in content


@pytest.mark.asyncio
async def test_memory_command_lists_populated_store(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """``/memory`` on a populated store enumerates the records."""
    from langgraph_kit.core.memory.models import (
        MemoryRecord,
        MemoryScope,
        MemoryType,
    )
    from langgraph_kit.core.memory.persistent import PersistentMemoryManager

    mgr = PersistentMemoryManager(e2e_store)
    await mgr.create(
        MemoryRecord(
            title="remembered fact",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
            summary="s",
            body="b",
        )
    )
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/memory",
        thread_id="cmd-memory-pop",
    )
    assert "remembered fact" in str(msg.content)


@pytest.mark.asyncio
async def test_compact_command_on_small_state_reports_no_compaction(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """``/compact`` on a small context reports 'No compaction needed'."""
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/compact",
        thread_id="cmd-compact-small",
    )
    content = str(msg.content)
    assert "No compaction needed" in content


@pytest.mark.asyncio
async def test_context_command_reports_window_pressure(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """``/context`` dispatches through to ``PressureMonitor.assess`` and
    surfaces the token / window / pressure fields in its output.

    Regression guard for the command ↔ monitor wiring: if a future
    refactor moves the pressure-monitor reference or changes its
    reported field shape, the ``/context`` handler silently produces a
    wrong (or empty) summary. Asserting on the structured keys here
    pins the contract.
    """
    msg = await _run_command(
        checkpointer, e2e_store, patched_build_llm,
        command="/context",
        thread_id="cmd-context",
    )
    content = str(msg.content)
    assert "Context Window Status" in content, (
        f"/context output should include the 'Context Window Status'"
        f" header; got {content!r}"
    )
    assert "Estimated tokens" in content, "Expected 'Estimated tokens' row"
    assert "Window limit" in content, "Expected 'Window limit' row"
    assert "Pressure" in content, "Expected 'Pressure' row"


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
