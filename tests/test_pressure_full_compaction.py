"""Tests for PressureMiddleware FULL_COMPACTION behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.context_management.pressure_middleware import (
    PressureMiddleware,
)

# A valid <summary> JSON block the CompactionPromptPack can parse.
_VALID_SUMMARY = """<analysis>working</analysis>
<summary>
{
  "user_intent": "find the bug",
  "key_decisions": ["use pytest", "skip e2e"],
  "important_files": ["src/foo.py"],
  "errors_and_fixes": ["KeyError fixed by guarding dict.get"],
  "current_state": "tests passing",
  "pending_work": ["write docs"],
  "next_step": "open PR"
}
</summary>"""


def _critical_pressure_messages(n: int = 150) -> list[HumanMessage]:
    """Build n small messages that collectively exceed 85% of the default window.

    Each message is kept below the 4000-token large-output threshold so the
    monitor prefers FULL_COMPACTION over MICROCOMPACT. 150 x 3000 chars ≈
    112k tokens ≈ 88% of the default 128k window.
    """
    blob = "x" * 3000
    return [HumanMessage(content=f"{blob} msg-{i}") for i in range(n)]


@pytest.mark.asyncio
async def test_full_compaction_happy_path_replaces_messages() -> None:
    """Critical pressure + valid LLM summary → messages replaced with summary + tail."""
    monitor = PressureMonitor()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))

    middleware = PressureMiddleware(monitor, llm=llm, compaction_tail_size=3)

    messages = _critical_pressure_messages()
    state = {"messages": messages}

    result = await middleware.abefore_agent(state, MagicMock())

    assert result is not None, "FULL_COMPACTION should have produced a new state"
    new_messages = result["messages"]
    # Summary + tail (3 messages) = 4 total
    assert len(new_messages) == 4
    assert "Conversation Summary" in new_messages[0].content
    assert "find the bug" in new_messages[0].content
    # Tail preserves last 3 originals
    assert new_messages[1:] == messages[-3:]
    # LLM was invoked exactly once
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_full_compaction_llm_failure_records_and_passes_through() -> None:
    """LLM raises → record_compaction_failure called, messages untouched."""
    monitor = PressureMonitor()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("llm exploded"))

    middleware = PressureMiddleware(monitor, llm=llm)

    messages = _critical_pressure_messages()
    state = {"messages": messages}

    result = await middleware.abefore_agent(state, MagicMock())

    assert result is None, "Failed compaction should not mutate state"
    # Circuit breaker should have registered the failure
    signals = monitor.assess(messages)
    assert signals.compaction_failures == 1


@pytest.mark.asyncio
async def test_full_compaction_without_llm_is_noop() -> None:
    """No LLM configured → FULL_COMPACTION branch returns None cleanly."""
    monitor = PressureMonitor()
    middleware = PressureMiddleware(monitor, llm=None)

    messages = _critical_pressure_messages()
    state = {"messages": messages}

    result = await middleware.abefore_agent(state, MagicMock())
    assert result is None


@pytest.mark.asyncio
async def test_full_compaction_tags_llm_call_as_internal() -> None:
    """Compactor's ainvoke must carry INTERNAL_TAG + CONTEXT_COMPACTION_TAG.

    Without these tags, the summary tokens stream back into the user's
    chat bubble alongside the agent's real reply.
    """
    from langgraph_kit.core.internal_tags import (
        CONTEXT_COMPACTION_TAG,
        INTERNAL_TAG,
    )

    monitor = PressureMonitor()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))

    middleware = PressureMiddleware(monitor, llm=llm, compaction_tail_size=3)

    messages = _critical_pressure_messages()
    await middleware.abefore_agent({"messages": messages}, MagicMock())

    llm.ainvoke.assert_awaited_once()
    config = llm.ainvoke.call_args.kwargs.get("config")
    assert config is not None, "compactor must pass config= to ainvoke"
    tags = config.get("tags", [])
    assert INTERNAL_TAG in tags
    assert CONTEXT_COMPACTION_TAG in tags
    assert config.get("run_name") == "context_compaction"
