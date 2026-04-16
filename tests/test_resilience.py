"""Tests for resilience middleware — error recovery, empty turns, completion guards."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------------------------
# 1. ToolErrorMiddleware
# ---------------------------------------------------------------------------
from langgraph_kit.core.resilience.tool_error import ToolErrorMiddleware


def _make_tool_request(name: str = "test_tool", call_id: str = "call_1") -> MagicMock:
    request = MagicMock()
    request.tool_call = {"name": name, "id": call_id}
    request.runtime = MagicMock()
    return request


@pytest.mark.asyncio
async def test_tool_error_catches_exception() -> None:
    mw = ToolErrorMiddleware(max_retries=0)
    request = _make_tool_request("broken_tool")
    handler = AsyncMock(side_effect=ValueError("bad input"))

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "broken_tool" in result.content
    assert "ValueError" in result.content
    assert "bad input" in result.content
    assert "retryable: False" in result.content


@pytest.mark.asyncio
async def test_tool_error_passes_through_on_success() -> None:
    mw = ToolErrorMiddleware(max_retries=0)
    request = _make_tool_request()
    expected = ToolMessage(content="success", tool_call_id="call_1")
    handler = AsyncMock(return_value=expected)

    result = await mw.awrap_tool_call(request, handler)

    assert result is expected


@pytest.mark.asyncio
async def test_tool_error_retries_retryable() -> None:
    mw = ToolErrorMiddleware(max_retries=1)
    request = _make_tool_request()

    # First call raises TimeoutError (retryable), second succeeds
    expected = ToolMessage(content="ok", tool_call_id="call_1")
    handler = AsyncMock(side_effect=[TimeoutError("timeout"), expected])

    result = await mw.awrap_tool_call(request, handler)
    assert result is expected
    assert handler.call_count == 2


@pytest.mark.asyncio
async def test_tool_error_no_retry_for_non_retryable() -> None:
    mw = ToolErrorMiddleware(max_retries=1)
    request = _make_tool_request()
    handler = AsyncMock(side_effect=TypeError("type error"))

    result = await mw.awrap_tool_call(request, handler)

    # Should NOT retry — TypeError is not in retryable set
    assert handler.call_count == 1
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


@pytest.mark.asyncio
async def test_tool_error_retries_exhausted() -> None:
    mw = ToolErrorMiddleware(max_retries=2)
    request = _make_tool_request()
    handler = AsyncMock(side_effect=TimeoutError("timeout"))

    result = await mw.awrap_tool_call(request, handler)

    # 1 initial + 2 retries = 3 calls
    assert handler.call_count == 3
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "retryable: True" in result.content


# ---------------------------------------------------------------------------
# 2. EmptyTurnMiddleware
# ---------------------------------------------------------------------------
from langgraph_kit.core.resilience.empty_turn import EmptyTurnMiddleware


@pytest.mark.asyncio
async def test_empty_turn_nudges_on_empty_output() -> None:
    mw = EmptyTurnMiddleware(max_nudges=2)
    state = {"messages": [AIMessage(content="")]}

    result = await mw.aafter_model(state, MagicMock())

    assert result is not None
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert "empty" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_empty_turn_allows_content() -> None:
    mw = EmptyTurnMiddleware()
    state = {"messages": [AIMessage(content="Here is the answer to your question.")]}

    result = await mw.aafter_model(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_empty_turn_allows_tool_calls() -> None:
    mw = EmptyTurnMiddleware()
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = [{"name": "search", "args": {}, "id": "tc_1"}]
    state = {"messages": [ai_msg]}

    result = await mw.aafter_model(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_empty_turn_respects_max_nudges() -> None:
    mw = EmptyTurnMiddleware(max_nudges=1)
    state = {"messages": [AIMessage(content="")]}

    # First nudge succeeds
    r1 = await mw.aafter_model(state, MagicMock())
    assert r1 is not None

    # Second nudge exceeds limit — allows termination
    r2 = await mw.aafter_model(state, MagicMock())
    assert r2 is None


@pytest.mark.asyncio
async def test_empty_turn_resets_on_valid_output() -> None:
    mw = EmptyTurnMiddleware(max_nudges=1)

    # First: empty
    empty_state = {"messages": [AIMessage(content="")]}
    r1 = await mw.aafter_model(empty_state, MagicMock())
    assert r1 is not None

    # Then: valid output resets counter
    valid_state = {"messages": [AIMessage(content="Here is a real response.")]}
    await mw.aafter_model(valid_state, MagicMock())

    # Now empty again — should nudge since counter was reset
    r3 = await mw.aafter_model(empty_state, MagicMock())
    assert r3 is not None


@pytest.mark.asyncio
async def test_empty_turn_whitespace_only_counts_as_empty() -> None:
    mw = EmptyTurnMiddleware()
    state = {"messages": [AIMessage(content="   \n  ")]}

    result = await mw.aafter_model(state, MagicMock())
    assert result is not None


@pytest.mark.asyncio
async def test_empty_turn_no_messages() -> None:
    mw = EmptyTurnMiddleware()
    result = await mw.aafter_model({"messages": []}, MagicMock())
    assert result is None


# ---------------------------------------------------------------------------
# 3. CompletionGuardMiddleware
# ---------------------------------------------------------------------------
from langgraph_kit.core.resilience.completion_guard import (
    CompletionGuardMiddleware,
)


def _build_conversation(
    *,
    tool_calls: int = 0,
    tool_errors: int = 0,
    final_content: str = "I've completed the task.",
) -> dict[str, list[Any]]:
    """Build a realistic conversation with configurable signals.

    Always produces at least _MIN_MESSAGES_BEFORE_GUARD messages so
    the guard doesn't skip due to conversation length.
    """
    messages: list[Any] = [
        HumanMessage(content="Please implement the feature."),
        AIMessage(content="Let me work on that."),
        HumanMessage(content="Yes, go ahead."),
        AIMessage(content="Understood, starting now."),
    ]

    for i in range(tool_calls):
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": f"tool_{i}", "args": {}, "id": f"tc_{i}"}]
        messages.append(ai)

        if i < tool_errors:
            messages.append(
                ToolMessage(
                    content="Error occurred", tool_call_id=f"tc_{i}", status="error"
                )
            )
        else:
            messages.append(ToolMessage(content="Success", tool_call_id=f"tc_{i}"))

    messages.append(AIMessage(content=final_content))
    return {"messages": messages}


@pytest.mark.asyncio
async def test_guard_triggers_on_completion_with_tool_error() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=0)
    state = _build_conversation(tool_calls=1, tool_errors=1)

    result = await mw.aafter_model(state, MagicMock())

    assert result is not None
    assert "premature" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_guard_triggers_on_no_tool_use() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=1)
    state = _build_conversation(tool_calls=0)

    result = await mw.aafter_model(state, MagicMock())

    assert result is not None
    assert "tool" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_guard_passes_clean_completion() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=1)
    state = _build_conversation(tool_calls=2, tool_errors=0)

    result = await mw.aafter_model(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_guard_ignores_non_completion_content() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=1)
    state = _build_conversation(
        tool_calls=0, final_content="Let me think about this more."
    )

    result = await mw.aafter_model(state, MagicMock())
    # Not a completion claim, so guard shouldn't trigger
    assert result is None


@pytest.mark.asyncio
async def test_guard_respects_max_challenges() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=1)
    state = _build_conversation(tool_calls=0)

    # First two challenges
    r1 = await mw.aafter_model(state, MagicMock())
    assert r1 is not None
    r2 = await mw.aafter_model(state, MagicMock())
    assert r2 is not None

    # Third exceeds _MAX_CHALLENGES (2)
    r3 = await mw.aafter_model(state, MagicMock())
    assert r3 is None


@pytest.mark.asyncio
async def test_guard_skips_short_conversations() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=1)
    # Only 2 messages — below _MIN_MESSAGES_BEFORE_GUARD
    state = {
        "messages": [
            HumanMessage(content="Do it"),
            AIMessage(content="Done, I've completed everything."),
        ]
    }

    result = await mw.aafter_model(state, MagicMock())
    assert result is None


# ---------------------------------------------------------------------------
# 4. PostRunBackstopMiddleware
# ---------------------------------------------------------------------------
from langgraph_kit.core.resilience.post_run import (
    PostRunBackstopMiddleware,
)


@pytest.mark.asyncio
async def test_post_run_records_metadata(mock_store: Any, monkeypatch: Any) -> None:
    mw = PostRunBackstopMiddleware()

    # Simulate run lifecycle
    await mw.abefore_agent({}, None)

    monkeypatch.setattr(
        "langgraph_kit.core.resilience.post_run.get_config",
        lambda: {"configurable": {"thread_id": "test-thread"}},
    )

    state = {
        "messages": [
            HumanMessage(content="Do something"),
            AIMessage(content="Done"),
        ]
    }

    class FakeRuntime:
        store = mock_store

    await mw.aafter_agent(state, FakeRuntime())

    # Verify metadata was persisted
    stored = mock_store._data.get(("run_metadata",))
    assert stored is not None
    assert len(stored) == 1
    metadata = next(iter(stored.values()))
    assert metadata["message_count"] == 2
    assert metadata["ai_messages"] == 1
    assert metadata["tool_calls"] == 0
    assert metadata["tool_errors"] == 0


@pytest.mark.asyncio
async def test_post_run_counts_tool_calls_and_errors(
    mock_store: Any, monkeypatch: Any
) -> None:
    mw = PostRunBackstopMiddleware()
    await mw.abefore_agent({}, None)

    monkeypatch.setattr(
        "langgraph_kit.core.resilience.post_run.get_config",
        lambda: {"configurable": {"thread_id": "test-thread-2"}},
    )

    ai_with_tools = AIMessage(content="")
    ai_with_tools.tool_calls = [
        {"name": "search", "args": {}, "id": "tc1"},
        {"name": "read", "args": {}, "id": "tc2"},
    ]

    state = {
        "messages": [
            HumanMessage(content="Find it"),
            ai_with_tools,
            ToolMessage(content="found", tool_call_id="tc1"),
            ToolMessage(content="error", tool_call_id="tc2", status="error"),
            AIMessage(content="Here are the results."),
        ]
    }

    class FakeRuntime:
        store = mock_store

    await mw.aafter_agent(state, FakeRuntime())

    stored = mock_store._data.get(("run_metadata",))
    metadata = next(iter(stored.values()))
    assert metadata["tool_calls"] == 2
    assert metadata["tool_errors"] == 1
    assert metadata["ai_messages"] == 2
    assert "results" in metadata["last_response_preview"]


@pytest.mark.asyncio
async def test_post_run_handles_no_store() -> None:
    mw = PostRunBackstopMiddleware()
    await mw.abefore_agent({}, None)

    class NoStoreRuntime:
        store = None

    # Should not raise
    await mw.aafter_agent({"messages": []}, NoStoreRuntime())
