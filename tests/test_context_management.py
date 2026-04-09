"""Tests for the context_management module — compaction, continuation, and pressure."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph_kit.core.context_management.compaction import (
    CompactionMode,
    CompactionPromptPack,
    CompactionResult,
)
from langgraph_kit.core.context_management.continuation import (
    ContinuationDecision,
    ContinuationTracker,
)
from langgraph_kit.core.context_management.pressure import (
    MitigationStrategy,
    PressureMonitor,
    PressureSignals,
)
from langgraph_kit.core.context_management.pressure_middleware import (
    PressureMiddleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMessage:
    def __init__(self, content: str, msg_type: str = "human") -> None:
        self.content = content
        self.type = msg_type


_VALID_RAW_OUTPUT = (
    "<analysis>Some analysis here</analysis>\n"
    "<summary>\n"
    '{"user_intent": "fix the bug", '
    '"key_decisions": ["use approach A"], '
    '"important_files": ["foo.py"], '
    '"errors_and_fixes": ["fixed null ref"], '
    '"current_state": "implementation done", '
    '"pending_work": ["run tests"], '
    '"next_step": "run pytest"}\n'
    "</summary>"
)


# ---------------------------------------------------------------------------
# CompactionPromptPack tests
# ---------------------------------------------------------------------------


class TestCompactionPromptPack:
    def test_build_prompt_full_mode(self) -> None:
        pack = CompactionPromptPack()
        prompt = pack.build_prompt(CompactionMode.FULL)
        assert "FULL" in prompt
        assert "Do NOT call any tools" in prompt

    def test_build_prompt_partial_mode(self) -> None:
        pack = CompactionPromptPack()
        prompt = pack.build_prompt(CompactionMode.PARTIAL)
        assert "PARTIAL" in prompt

    def test_build_prompt_with_session_notebook(self) -> None:
        pack = CompactionPromptPack()
        notebook_text = "## Notes\n- item one\n- item two"
        prompt = pack.build_prompt(CompactionMode.FULL, session_notebook=notebook_text)
        assert notebook_text in prompt
        assert "Session Notebook" in prompt

    def test_parse_output_valid(self) -> None:
        pack = CompactionPromptPack()
        result = pack.parse_output(_VALID_RAW_OUTPUT, mode=CompactionMode.FULL)
        assert result is not None
        assert isinstance(result, CompactionResult)
        assert result.user_intent == "fix the bug"
        assert result.key_decisions == ["use approach A"]
        assert result.important_files == ["foo.py"]
        assert result.errors_and_fixes == ["fixed null ref"]
        assert result.current_state == "implementation done"
        assert result.pending_work == ["run tests"]
        assert result.next_step == "run pytest"
        assert result.mode == CompactionMode.FULL

    def test_parse_output_invalid(self) -> None:
        pack = CompactionPromptPack()
        result = pack.parse_output("<summary>not valid json</summary>")
        assert result is None

    def test_parse_output_no_summary_tag(self) -> None:
        pack = CompactionPromptPack()
        result = pack.parse_output("just some text without tags")
        assert result is None

    def test_parse_analysis(self) -> None:
        pack = CompactionPromptPack()
        analysis = pack.parse_analysis(_VALID_RAW_OUTPUT)
        assert analysis == "Some analysis here"

    def test_parse_analysis_missing(self) -> None:
        pack = CompactionPromptPack()
        analysis = pack.parse_analysis("no analysis tags here")
        assert analysis == ""


# ---------------------------------------------------------------------------
# ContinuationTracker tests
# ---------------------------------------------------------------------------


class TestContinuationTracker:
    def test_initial_should_continue(self) -> None:
        tracker = ContinuationTracker(budget_tokens=100_000)
        decision = tracker.should_continue()
        assert decision.action == "continue"
        assert decision.budget_consumed_pct == 0.0
        assert decision.continuation_count == 0

    def test_stop_at_budget_threshold(self) -> None:
        tracker = ContinuationTracker(budget_tokens=1000, stop_threshold_pct=0.90)
        # Record enough tokens to exceed 90% of the 1000-token budget
        tracker.record_turn(950)
        decision = tracker.should_continue()
        assert decision.action == "stop"
        assert "Budget" in decision.reason or "budget" in decision.reason.lower()

    def test_stop_at_max_continuations(self) -> None:
        tracker = ContinuationTracker(budget_tokens=1_000_000, max_continuations=3)
        for _ in range(3):
            tracker.record_turn(10)
        decision = tracker.should_continue()
        assert decision.action == "stop"
        assert "Max" in decision.reason or "max" in decision.reason.lower()

    def test_diminishing_returns_detected(self) -> None:
        tracker = ContinuationTracker(
            budget_tokens=1_000_000,
            max_continuations=100,
            diminishing_returns_ratio=0.3,
            min_turns_for_dr=3,
        )
        # Large early turns
        tracker.record_turn(5000)
        tracker.record_turn(5000)
        # Tiny recent turns — average 100 vs earlier average 5000 => ratio 0.02
        tracker.record_turn(100)
        tracker.record_turn(100)
        decision = tracker.should_continue()
        assert decision.action == "stop"
        assert decision.diminishing_returns is True

    def test_no_diminishing_returns_with_few_turns(self) -> None:
        tracker = ContinuationTracker(
            budget_tokens=1_000_000,
            min_turns_for_dr=5,
        )
        tracker.record_turn(5000)
        tracker.record_turn(10)
        decision = tracker.should_continue()
        # Only 2 turns, min is 5 — DR should not trigger
        assert decision.diminishing_returns is False

    def test_reset(self) -> None:
        tracker = ContinuationTracker(budget_tokens=10_000)
        tracker.record_turn(5000)
        tracker.record_turn(5000)
        tracker.reset()
        decision = tracker.should_continue()
        assert decision.action == "continue"
        assert decision.total_tokens_used == 0
        assert decision.continuation_count == 0

    def test_decision_metadata(self) -> None:
        tracker = ContinuationTracker(budget_tokens=10_000)
        tracker.record_turn(1000)
        decision = tracker.should_continue()
        assert isinstance(decision, ContinuationDecision)
        assert decision.action in ("continue", "stop")
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0
        assert decision.budget_consumed_pct == pytest.approx(0.1)
        assert decision.continuation_count == 1
        assert decision.total_tokens_used == 1000
        assert isinstance(decision.diminishing_returns, bool)


# ---------------------------------------------------------------------------
# PressureMonitor tests
# ---------------------------------------------------------------------------


class TestPressureMonitor:
    def test_assess_empty_messages(self) -> None:
        monitor = PressureMonitor()
        signals = monitor.assess([])
        assert signals.estimated_tokens == 0
        assert signals.pressure_pct == 0.0

    def test_assess_estimates_tokens(self) -> None:
        monitor = PressureMonitor(window_limit=128_000)
        # A message with 400 chars => ~100 tokens (len // 4)
        msg = MockMessage("a" * 400)
        signals = monitor.assess([msg])
        assert signals.estimated_tokens == 100

    def test_no_mitigation_low_pressure(self) -> None:
        monitor = PressureMonitor(window_limit=128_000, warn_pct=0.70)
        signals = PressureSignals(
            estimated_tokens=1000,
            window_limit=128_000,
            pressure_pct=0.10,
            large_tool_outputs=0,
            compaction_failures=0,
        )
        strategy = monitor.choose_mitigation(signals)
        assert strategy == MitigationStrategy.NONE

    def test_microcompact_moderate_pressure_with_large_outputs(self) -> None:
        monitor = PressureMonitor(
            window_limit=128_000, warn_pct=0.70, critical_pct=0.85
        )
        signals = PressureSignals(
            estimated_tokens=100_000,
            window_limit=128_000,
            pressure_pct=0.78,
            large_tool_outputs=5,
            compaction_failures=0,
        )
        strategy = monitor.choose_mitigation(signals)
        assert strategy == MitigationStrategy.MICROCOMPACT

    def test_full_compaction_critical_pressure(self) -> None:
        monitor = PressureMonitor(
            window_limit=128_000, warn_pct=0.70, critical_pct=0.85
        )
        signals = PressureSignals(
            estimated_tokens=115_000,
            window_limit=128_000,
            pressure_pct=0.90,
            large_tool_outputs=0,
            compaction_failures=0,
        )
        strategy = monitor.choose_mitigation(signals)
        assert strategy == MitigationStrategy.FULL_COMPACTION

    def test_circuit_breaker(self) -> None:
        monitor = PressureMonitor(max_compaction_failures=3)
        for _ in range(3):
            monitor.record_compaction_failure()
        signals = PressureSignals(
            estimated_tokens=100_000,
            window_limit=128_000,
            pressure_pct=0.90,
            large_tool_outputs=0,
            compaction_failures=3,
        )
        strategy = monitor.choose_mitigation(signals)
        assert strategy == MitigationStrategy.STOP

    def test_record_compaction_success_resets_failures(self) -> None:
        monitor = PressureMonitor(max_compaction_failures=3)
        monitor.record_compaction_failure()
        monitor.record_compaction_failure()
        monitor.record_compaction_success()
        # After success, the circuit breaker should be reset
        signals = PressureSignals(
            estimated_tokens=115_000,
            window_limit=128_000,
            pressure_pct=0.90,
            large_tool_outputs=0,
            compaction_failures=0,
        )
        strategy = monitor.choose_mitigation(signals)
        # Should NOT be STOP since failures were reset
        assert strategy != MitigationStrategy.STOP


# ---------------------------------------------------------------------------
# PressureMiddleware tests (async)
# ---------------------------------------------------------------------------


class _CopyableMessage(MockMessage):
    """MockMessage that supports model_copy for PressureMiddleware tests."""

    def model_copy(self, *, update: dict[str, Any] | None = None) -> _CopyableMessage:
        new = _CopyableMessage(self.content, self.type)
        if update:
            for k, v in update.items():
                setattr(new, k, v)
        return new


class TestPressureMiddleware:
    @pytest.mark.asyncio
    async def test_before_agent_no_pressure(self) -> None:
        monitor = PressureMonitor(window_limit=128_000)
        middleware = PressureMiddleware(monitor=monitor)
        state: dict[str, list[MockMessage]] = {
            "messages": [MockMessage("hello", "human")],
        }
        result = await middleware.abefore_agent(state)
        # No pressure — should return None (no state update)
        assert result is None

    @pytest.mark.asyncio
    async def test_microcompact_truncates_old_tool_outputs(self) -> None:
        monitor = PressureMonitor(
            window_limit=1000,
            warn_pct=0.30,
            critical_pct=0.60,
            large_output_threshold=50,
        )
        middleware = PressureMiddleware(monitor=monitor)

        old_tool_messages = [
            _CopyableMessage("x" * 3000, msg_type="tool") for _ in range(5)
        ]
        recent_messages = [
            _CopyableMessage("short", msg_type="human") for _ in range(10)
        ]
        all_messages: list[Any] = [*old_tool_messages, *recent_messages]
        state: dict[str, Any] = {"messages": all_messages}

        result = await middleware.abefore_agent(state)

        # Should return a state update with compacted messages
        assert result is not None
        new_messages = result["messages"]

        # Old tool messages (indices 0-4) should have been truncated
        for msg in new_messages[:5]:
            assert "truncated" in msg.content
            assert len(msg.content) < 3000

        # Recent messages (last 10) should remain intact
        for msg in new_messages[-10:]:
            assert msg.content == "short"
