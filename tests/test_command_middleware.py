"""Tests for CommandMiddleware — slash-command interception and short-circuit."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.core.commands.dispatch import CommandDispatcher, CommandResult
from langgraph_kit.core.commands.middleware import CommandMiddleware


@pytest.fixture
def dispatcher() -> CommandDispatcher:
    return CommandDispatcher()


@pytest.mark.asyncio
async def test_non_command_input_passes_through(
    dispatcher: CommandDispatcher,
) -> None:
    """Plain user text must not trigger the dispatcher at all."""
    mw = CommandMiddleware(dispatcher)
    state = {"messages": [HumanMessage(content="hello")]}
    result = await mw.abefore_agent(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_unregistered_slash_command_passes_through(
    dispatcher: CommandDispatcher,
) -> None:
    """Unknown ``/command`` input must not short-circuit — the agent still gets it."""
    mw = CommandMiddleware(dispatcher)
    state = {"messages": [HumanMessage(content="/unknown_command")]}
    result = await mw.abefore_agent(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_handled_command_emits_jump_to_end(
    dispatcher: CommandDispatcher,
) -> None:
    """On a handled command the middleware MUST return ``jump_to: "end"``.

    Regression test for the double-response bug: a prior version returned
    only ``{"messages": [AIMessage]}`` from ``abefore_agent``, which
    appended the command output but let the model node run immediately
    after — so the user saw both the command result AND an LLM response
    continuing the conversation. Short-circuiting requires ``jump_to``,
    and LangChain's wiring only honors it when the hook is decorated with
    ``@hook_config(can_jump_to=["end"])``.
    """
    handler = AsyncMock(return_value=CommandResult(output="help text"))
    dispatcher.register("help", handler)

    mw = CommandMiddleware(dispatcher)
    state = {"messages": [HumanMessage(content="/help")]}
    result = await mw.abefore_agent(state, MagicMock())

    assert result is not None
    assert result["jump_to"] == "end", (
        "CommandMiddleware must short-circuit via jump_to=end, otherwise the "
        "model runs after the command output and produces a duplicate reply"
    )
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "help text"


@pytest.mark.asyncio
async def test_handled_command_preserves_user_message_via_add_messages_reducer(
    dispatcher: CommandDispatcher,
) -> None:
    """The return value is merged via the ``add_messages`` reducer, not
    assigned. The middleware returns only the new AI message — the reducer
    appends it to the existing list, so the user's original message remains.

    This test documents the intended reducer behavior so a future change
    to plain-list state doesn't silently drop the user's message.
    """
    handler = AsyncMock(return_value=CommandResult(output="hi"))
    dispatcher.register("greet", handler)

    mw = CommandMiddleware(dispatcher)
    state = {"messages": [HumanMessage(content="/greet")]}
    result = await mw.abefore_agent(state, MagicMock())

    assert result is not None
    # Middleware contributes ONLY the new AI message — it MUST NOT try to
    # repeat the existing user message in the update (that would double it
    # under add_messages).
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_compact_command_merges_replacement_messages(
    dispatcher: CommandDispatcher,
) -> None:
    """``/compact`` returns a replacement transcript via ``compacted_messages``.

    The middleware forwards those messages along with the summary AI
    message AND ``jump_to: "end"``. Messages with the same id as existing
    ones replace in place via ``add_messages``; the new AI summary is
    appended.
    """
    original_a = HumanMessage(content="old msg", id="msg-a")
    original_a_truncated = HumanMessage(content="old msg (truncated)", id="msg-a")

    handler = AsyncMock(
        return_value=CommandResult(
            output="compacted",
            metadata={"compacted_messages": [original_a_truncated]},
        )
    )
    dispatcher.register("compact", handler)

    mw = CommandMiddleware(dispatcher)
    state = {"messages": [original_a, HumanMessage(content="/compact")]}
    result = await mw.abefore_agent(state, MagicMock())

    assert result is not None
    assert result["jump_to"] == "end"
    # compacted messages precede the summary AI message
    assert result["messages"][0] is original_a_truncated
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "compacted"


def test_abefore_agent_is_declared_jumpable() -> None:
    """The ``@hook_config(can_jump_to=["end"])`` decorator MUST be present.

    Without the ``__can_jump_to__`` attribute on the hook, LangChain's
    graph wiring silently ignores any ``jump_to`` the middleware emits,
    and the model runs anyway — producing the double-response bug.
    """
    jumpable: list[Any] = getattr(
        CommandMiddleware.abefore_agent, "__can_jump_to__", []
    )
    assert "end" in jumpable, (
        "abefore_agent must be decorated with @hook_config(can_jump_to=['end']) "
        "so LangChain honors the jump_to short-circuit"
    )
