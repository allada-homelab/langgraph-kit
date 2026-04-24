"""Tests for PressureMiddleware FULL_COMPACTION behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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
    # RemoveMessage(REMOVE_ALL) + summary + tail (3 messages) = 5 total.
    # The REMOVE_ALL marker makes the replacement intent explicit to the
    # add_messages reducer; without it, a non-standard reducer would silently
    # append the summary instead of replacing old messages.
    assert len(new_messages) == 5
    assert isinstance(new_messages[0], RemoveMessage)
    assert new_messages[0].id == REMOVE_ALL_MESSAGES
    assert "Conversation Summary" in new_messages[1].content
    assert "find the bug" in new_messages[1].content
    # Tail preserves last 3 originals (after the summary).
    assert new_messages[2:] == messages[-3:]
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


@pytest.mark.asyncio
async def test_full_compaction_replacement_survives_add_messages_reducer() -> None:
    """Applying the middleware output through ``add_messages`` must actually
    replace the old messages — not append the summary on top of them.

    Earlier versions returned only ``[summary, *tail]``; without a
    ``RemoveMessage`` marker the reducer treated the summary as a new message
    and left every old message in place, making compaction a no-op in real
    langgraph state.
    """
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage as _HumanMessage,
    )
    from langgraph.graph.message import add_messages

    monitor = PressureMonitor()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))
    middleware = PressureMiddleware(monitor, llm=llm, compaction_tail_size=3)

    # Need ids on messages for add_messages; use smaller pressure sample.
    messages = [_HumanMessage(content="x" * 3000, id=f"m{i}") for i in range(150)]
    state = {"messages": messages}

    result = await middleware.abefore_agent(state, MagicMock())
    assert result is not None

    # ``add_messages`` is typed as ``Messages | Callable[..., Messages]``
    # because it doubles as a reducer-factory. When called with both args
    # it always returns the merged list, but basedpyright can't narrow the
    # union from a single call site — suppress at the seam so the
    # assertions read naturally.
    merged: list[Any] = add_messages(  # pyright: ignore[reportAssignmentType]
        messages,  # pyright: ignore[reportArgumentType]
        result["messages"],
    )
    # After merging, expect only the summary + tail (3) = 4 messages. If the
    # old messages had leaked through, len(merged) would still be ~150.
    assert len(merged) == 4
    assert "Conversation Summary" in str(merged[0].content)
    assert merged[1:] == messages[-3:]


@pytest.mark.asyncio
async def test_partial_compaction_runs_when_enabled_at_moderate_pressure() -> None:
    """With ``enable_partial_compaction=True`` and moderate pressure, the
    middleware runs a PARTIAL summarization instead of doing nothing."""
    monitor = PressureMonitor(
        window_limit=128_000,
        warn_pct=0.70,
        critical_pct=0.85,
        enable_partial_compaction=True,
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))
    middleware = PressureMiddleware(
        monitor, llm=llm, partial_keep_size=5, compaction_tail_size=3
    )

    # Moderate pressure: 75 x 3000 chars ≈ 56k tokens ≈ 44% — below warn.
    # Use 110 messages to land in the moderate band without large outputs.
    messages = [HumanMessage(content="x" * 3000, id=f"m{i}") for i in range(110)]
    # 110 * 3000 / 4 ≈ 82500 tokens → ~64% — still below warn_pct.
    # Bump up to 120 messages to cross 70% but stay below 85%.
    messages = [HumanMessage(content="x" * 3000, id=f"m{i}") for i in range(120)]
    # 120 * 3000 / 4 = 90000 tokens → 70.3% → moderate band.

    result = await middleware.abefore_agent({"messages": messages}, MagicMock())
    assert result is not None, "PARTIAL_COMPACTION should have produced output"
    new_messages = result["messages"]
    # RemoveMessage + summary + 5 tail = 7
    assert isinstance(new_messages[0], RemoveMessage)
    assert new_messages[0].id == REMOVE_ALL_MESSAGES
    assert "Conversation Summary" in new_messages[1].content
    assert new_messages[2:] == messages[-5:]
    llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_compaction_disabled_by_default() -> None:
    """Default config keeps the old behavior — no LLM call at moderate pressure
    without large tool outputs."""
    monitor = PressureMonitor(window_limit=128_000, warn_pct=0.70, critical_pct=0.85)
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))
    middleware = PressureMiddleware(monitor, llm=llm)

    # 120 messages → ~70% pressure, no large outputs.
    messages = [HumanMessage(content="x" * 3000, id=f"m{i}") for i in range(120)]
    result = await middleware.abefore_agent({"messages": messages}, MagicMock())
    assert result is None
    llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_compaction_chunked_path_triggers_for_very_large_head() -> None:
    """When the rendered head would blow past the chunk threshold, the
    middleware summarizes chunk-by-chunk and does a final reduce — peak memory
    stays bounded to one chunk + per-chunk summaries."""
    monitor = PressureMonitor()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=_VALID_SUMMARY))

    middleware = PressureMiddleware(
        monitor,
        llm=llm,
        compaction_tail_size=3,
        # Force chunking: threshold below one single message's content size.
        chunk_render_threshold=1_000,
        chunk_messages=20,
    )

    messages = _critical_pressure_messages(n=150)
    result = await middleware.abefore_agent({"messages": messages}, MagicMock())
    assert result is not None

    # head has 147 messages, chunk size 20 → 8 chunks → 8 map + 1 reduce = 9 calls.
    expected_chunks = -(-(150 - 3) // 20)  # ceil division
    assert llm.ainvoke.await_count == expected_chunks + 1


def test_circuit_breaker_resets_after_cooldown_elapses() -> None:
    """A long-running session that tripped the breaker early should be able to
    recover once the cooldown passes — previously the breaker stayed open
    until process restart."""
    monitor = PressureMonitor(
        max_compaction_failures=3,
        compaction_cooldown_seconds=60.0,
    )
    for _ in range(3):
        monitor.record_compaction_failure()

    from langgraph_kit.core.context_management.pressure import (
        MitigationStrategy,
        PressureSignals,
    )

    critical = PressureSignals(
        estimated_tokens=115_000,
        window_limit=128_000,
        pressure_pct=0.90,
        large_tool_outputs=0,
        compaction_failures=3,
    )
    # Still inside cooldown.
    assert monitor.choose_mitigation(critical) == MitigationStrategy.STOP

    # Fast-forward the breaker past the cooldown by adjusting the internal
    # timestamp — exercising the public behavior (auto-reset after cooldown)
    # without relying on wall clock.
    import time as _time

    monitor._breaker_opened_at = _time.monotonic() - 61.0  # pyright: ignore[reportPrivateUsage]
    # Should auto-reset and allow compaction again.
    assert monitor.choose_mitigation(critical) == MitigationStrategy.FULL_COMPACTION


def test_pluggable_token_estimator_is_used() -> None:
    """Custom tokenizer replaces the len//4 heuristic for consumers that need
    billing-accurate counts (e.g. tiktoken, provider SDKs)."""
    calls: list[str] = []

    def count(text: str) -> int:
        calls.append(text)
        return 7  # fixed for assertion

    monitor = PressureMonitor(window_limit=100, token_estimator=count)

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    signals = monitor.assess([_Msg("hello"), _Msg("world")])
    assert signals.estimated_tokens == 14  # 2 * 7
    assert calls == ["hello", "world"]
