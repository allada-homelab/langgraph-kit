"""Tests for AutoMemoryExtractor and ExtractionMiddleware."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure langchain_core.messages is importable even without the real package.
# The extractor does `from langchain_core.messages import HumanMessage` inside
# its extract() method, so we stub the module if it's missing.
if "langchain_core" not in sys.modules:
    _lc = MagicMock()
    sys.modules["langchain_core"] = _lc
    sys.modules["langchain_core.messages"] = _lc.messages
    _lc.messages.HumanMessage = MagicMock

from langgraph_kit.core.memory.extraction import AutoMemoryExtractor
from langgraph_kit.core.memory.extraction_middleware import (
    ExtractionMiddleware,
    _agent_wrote_memory,
)
from langgraph_kit.core.memory.models import MemoryScope
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

# ---------------------------------------------------------------------------
# Fixtures — MockStore comes from conftest.py
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_manager(mock_store: Any) -> PersistentMemoryManager:
    return PersistentMemoryManager(mock_store)


def _make_llm_mock(content: str) -> AsyncMock:
    """Return an AsyncMock LLM whose ainvoke returns a message with *content*."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=content)
    return mock_llm


# ---------------------------------------------------------------------------
# AutoMemoryExtractor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_skips_when_agent_wrote_memory(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Returns empty when agent_wrote_memory_this_turn=True."""
    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)

    result = await extractor.extract(
        recent_messages=[MagicMock(type="human", content="hello")],
        agent_wrote_memory_this_turn=True,
    )

    assert result == []
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_extract_skips_empty_messages(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Returns empty when no messages are provided."""
    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)

    result = await extractor.extract(recent_messages=[])

    assert result == []
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_extract_creates_memory_from_llm_response(
    memory_manager: PersistentMemoryManager,
    mock_store: Any,
) -> None:
    """Mock the LLM to return a JSON array with one create action."""
    llm_response = (
        '[{"action": "create", "title": "User prefers Python", '
        '"type": "user", "scope": "user", '
        '"summary": "User likes Python", '
        '"body": "The user prefers Python for backend work."}]'
    )
    llm = _make_llm_mock(llm_response)
    extractor = AutoMemoryExtractor(memory_manager, llm)

    messages = [MagicMock(type="human", content="I really prefer Python for backend")]
    result = await extractor.extract(recent_messages=messages)

    assert len(result) == 1
    assert result[0].title == "User prefers Python"
    assert result[0].body == "The user prefers Python for backend work."
    assert result[0].source == "auto_extraction"

    # Verify the record was persisted in the store
    saved = await memory_manager.get(result[0].id, MemoryScope.USER)
    assert saved is not None
    assert saved.title == "User prefers Python"


@pytest.mark.asyncio
async def test_extract_handles_llm_failure(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Mock LLM to raise exception, verify returns empty list."""
    llm = AsyncMock()
    llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
    extractor = AutoMemoryExtractor(memory_manager, llm)

    result = await extractor.extract(
        recent_messages=[MagicMock(type="human", content="hello")]
    )

    assert result == []


@pytest.mark.asyncio
async def test_extract_handles_malformed_response(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Mock LLM to return garbage text, verify returns empty."""
    llm = _make_llm_mock("This is not JSON at all. }{][garbage")
    extractor = AutoMemoryExtractor(memory_manager, llm)

    result = await extractor.extract(
        recent_messages=[MagicMock(type="human", content="hello")]
    )

    assert result == []


@pytest.mark.asyncio
async def test_extract_skips_invalid_type_and_keeps_rest(
    memory_manager: PersistentMemoryManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An invalid ``type`` value on one candidate must not kill the batch.

    Regression test for the production crash: the extractor LLM returned a
    candidate with ``"type": "assistant"`` (not a MemoryType member); the
    enum constructor raised ValueError and the surrounding per-candidate
    try/except logged a traceback that looked like a hard failure. With
    pre-validation via ``coerce_memory_type``, the bad candidate is dropped
    at WARN (no traceback) and sibling candidates still persist.
    """
    llm_response = """[
        {
            "action": "create",
            "title": "Available Tools and Operational Constraints",
            "type": "assistant",
            "scope": "assistant",
            "summary": "Captures current toolset",
            "body": "Available tools: ls, read_file, ..."
        },
        {
            "action": "create",
            "title": "User prefers TDD",
            "type": "feedback",
            "scope": "user",
            "summary": "Test-first workflow requested",
            "body": "Write failing tests before implementing features."
        }
    ]"""
    llm = _make_llm_mock(llm_response)
    extractor = AutoMemoryExtractor(memory_manager, llm)

    with caplog.at_level("WARNING", logger="langgraph_kit.core.memory.extraction"):
        result = await extractor.extract(
            recent_messages=[MagicMock(type="human", content="hi")]
        )

    # Only the valid candidate persisted.
    assert len(result) == 1
    assert result[0].title == "User prefers TDD"

    # The invalid one was reported at WARN — not as a traceback-bearing
    # ``logger.exception`` — and named the offending value so the operator
    # can diagnose the extractor prompt.
    warning_texts = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("assistant" in text for text in warning_texts), warning_texts
    # And no ERROR-level noise from an unhandled enum ValueError.
    assert not any(r.levelname == "ERROR" for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


@pytest.mark.asyncio
async def test_extract_tags_llm_call_with_internal_tag(
    memory_manager: PersistentMemoryManager,
) -> None:
    """The extractor's ainvoke must carry ``langgraph_kit:internal`` and
    ``langgraph_kit:memory_extraction`` tags so consumers can filter the
    resulting chat_model events out of the user-facing SSE stream.

    Regression test for the leak bug: without these tags, the extractor's
    JSON output would stream back to the user's chat bubble after the main
    agent reply finished.
    """
    from langgraph_kit.core.internal_tags import (
        INTERNAL_TAG,
        MEMORY_EXTRACTION_TAG,
    )

    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)

    await extractor.extract(recent_messages=[MagicMock(type="human", content="hi")])

    llm.ainvoke.assert_awaited_once()
    call_kwargs = llm.ainvoke.call_args.kwargs
    config = call_kwargs.get("config")
    assert config is not None, "ainvoke must be called with a config= kwarg"
    tags = config.get("tags", [])
    assert INTERNAL_TAG in tags
    assert MEMORY_EXTRACTION_TAG in tags
    # run_name is visible on astream_events; useful for observability too.
    assert config.get("run_name") == "memory_extraction"


# ---------------------------------------------------------------------------
# ExtractionMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_before_agent_resets_tracking(
    memory_manager: PersistentMemoryManager,
) -> None:
    """After awrap_tool_call with save_memory, abefore_agent resets tracking."""
    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)
    middleware = ExtractionMiddleware(extractor)

    # Simulate a save_memory tool call to set the flag
    mock_request = MagicMock()
    mock_request.tool_call = {"name": "save_memory"}
    mock_handler = AsyncMock(return_value=MagicMock())
    await middleware.awrap_tool_call(mock_request, mock_handler)
    assert _agent_wrote_memory.get() is True

    # abefore_agent should reset
    await middleware.abefore_agent({}, None)
    assert _agent_wrote_memory.get() is False


@pytest.mark.asyncio
async def test_wrap_tool_call_tracks_memory_writes(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Calling save_memory sets the flag."""
    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)
    middleware = ExtractionMiddleware(extractor)

    mock_request = MagicMock()
    mock_request.tool_call = {"name": "save_memory"}
    mock_handler = AsyncMock(return_value=MagicMock())
    await middleware.awrap_tool_call(mock_request, mock_handler)

    assert _agent_wrote_memory.get() is True


@pytest.mark.asyncio
async def test_wrap_tool_call_ignores_other_tools(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Calling read_file does NOT set the flag."""
    llm = _make_llm_mock("[]")
    extractor = AutoMemoryExtractor(memory_manager, llm)
    middleware = ExtractionMiddleware(extractor)

    mock_request = MagicMock()
    mock_request.tool_call = {"name": "read_file"}
    mock_handler = AsyncMock(return_value=MagicMock())
    await middleware.awrap_tool_call(mock_request, mock_handler)

    assert _agent_wrote_memory.get() is False


@pytest.mark.asyncio
async def test_wrap_model_call_runs_extraction(
    memory_manager: PersistentMemoryManager,
) -> None:
    """Mock extractor, verify extract() called via aafter_agent."""
    mock_extractor = AsyncMock(spec=AutoMemoryExtractor)
    mock_extractor.extract.return_value = []

    middleware = ExtractionMiddleware(mock_extractor)

    messages = [MagicMock(type="human", content="hello")]
    state = {"messages": messages}

    await middleware.aafter_agent(state, None)

    mock_extractor.extract.assert_awaited_once()
    call_kwargs = mock_extractor.extract.call_args
    assert call_kwargs.kwargs["agent_wrote_memory_this_turn"] is False
