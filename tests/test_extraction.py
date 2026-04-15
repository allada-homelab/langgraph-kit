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
