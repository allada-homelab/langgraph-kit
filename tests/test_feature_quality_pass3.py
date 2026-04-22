"""Tests for pass-3 feature-quality improvements.

Covers:
- HITL ``_format_response`` — renders HumanResponse-like payloads as strings
- UI event tool input validation
- Replay ``assert_no_errors`` and ``assert_output_similarity`` using structured status
- ContinuationTracker sliding-window DR detection
"""

from __future__ import annotations

import pytest

from langgraph_kit.core.context_management.continuation import ContinuationTracker
from langgraph_kit.core.hitl.tools import _format_response
from langgraph_kit.core.ui_events import (
    CITATION_SENTINEL,
    PROGRESS_SENTINEL,
    SUGGESTIONS_SENTINEL,
    build_citation_tool,
    build_progress_tool,
    build_suggestions_tool,
)
from langgraph_kit.replay.assertions import ReplayAssertions
from langgraph_kit.replay.models import (
    ConversationRecording,
    LLMInteraction,
    ToolInteraction,
)

# ---------------------------------------------------------------------------
# HITL _format_response
# ---------------------------------------------------------------------------


class TestFormatResponse:
    def test_accept(self) -> None:
        assert _format_response({"type": "accept"}) == "User accepted the action."

    def test_ignore(self) -> None:
        result = _format_response({"type": "ignore"})
        assert "ignored" in result.lower()

    def test_response_with_message(self) -> None:
        result = _format_response({"type": "response", "args": "too risky"})
        assert "rejected" in result.lower()
        assert "too risky" in result

    def test_response_without_message(self) -> None:
        result = _format_response({"type": "response"})
        assert "rejected" in result.lower()

    def test_edit(self) -> None:
        result = _format_response(
            {"type": "edit", "args": {"path": "/etc/config.yaml"}}
        )
        assert "edit" in result.lower()
        assert "/etc/config.yaml" in result

    def test_list_wrapping(self) -> None:
        assert _format_response([{"type": "accept"}]) == "User accepted the action."

    def test_empty_list(self) -> None:
        # Empty list falls through to "User response:" with the empty dict repr.
        result = _format_response([])
        assert "response" in result.lower()

    def test_non_dict(self) -> None:
        assert _format_response("raw string") == "User response: raw string"

    def test_unknown_type(self) -> None:
        result = _format_response({"type": "weird"})
        assert "response" in result.lower()


# ---------------------------------------------------------------------------
# UI event validation
# ---------------------------------------------------------------------------


class TestProgressValidation:
    @pytest.mark.asyncio
    async def test_valid(self) -> None:
        tool = build_progress_tool()
        out = await tool("Searching", 1, 3)
        assert out.startswith(PROGRESS_SENTINEL)

    @pytest.mark.asyncio
    async def test_current_exceeds_total(self) -> None:
        tool = build_progress_tool()
        out = await tool("Step", 5, 3)
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_total_zero(self) -> None:
        tool = build_progress_tool()
        out = await tool("Step", 1, 0)
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_current_zero(self) -> None:
        tool = build_progress_tool()
        out = await tool("Step", 0, 3)
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_empty_step(self) -> None:
        tool = build_progress_tool()
        out = await tool("   ", 1, 3)
        assert out.startswith("Error")


class TestSuggestionsValidation:
    @pytest.mark.asyncio
    async def test_valid(self) -> None:
        tool = build_suggestions_tool()
        out = await tool(["Deploy", "Review"])
        assert out.startswith(SUGGESTIONS_SENTINEL)

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        tool = build_suggestions_tool()
        out = await tool([])
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_all_whitespace(self) -> None:
        tool = build_suggestions_tool()
        out = await tool(["", "   "])
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_too_many(self) -> None:
        tool = build_suggestions_tool()
        out = await tool([f"a{i}" for i in range(10)])
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_long_label_truncated(self) -> None:
        import json

        tool = build_suggestions_tool()
        out = await tool(["x" * 200])
        assert out.startswith(SUGGESTIONS_SENTINEL)
        payload = json.loads(out[len(SUGGESTIONS_SENTINEL) :])
        assert payload["actions"][0].endswith("…")
        assert len(payload["actions"][0]) < 200


class TestCitationValidation:
    @pytest.mark.asyncio
    async def test_valid(self) -> None:
        tool = build_citation_tool()
        out = await tool("auth.py:42", "src/auth.py")
        assert out.startswith(CITATION_SENTINEL)

    @pytest.mark.asyncio
    async def test_empty_title(self) -> None:
        tool = build_citation_tool()
        out = await tool("", "src/auth.py")
        assert out.startswith("Error")

    @pytest.mark.asyncio
    async def test_empty_source(self) -> None:
        tool = build_citation_tool()
        out = await tool("auth.py", "")
        assert out.startswith("Error")


# ---------------------------------------------------------------------------
# Replay assertions
# ---------------------------------------------------------------------------


def _rec_with(
    llm_content: str = "", tool_status: str = "success"
) -> ConversationRecording:
    return ConversationRecording(
        interactions=[
            ToolInteraction(
                sequence_num=1,
                tool_name="search",
                tool_output="result",
                status=tool_status,  # type: ignore[arg-type]
            ),
            LLMInteraction(
                sequence_num=2,
                output_message={"content": llm_content},
            ),
        ]
    )


class TestAssertNoErrorsStructured:
    def test_no_errors_ignores_output_text(self) -> None:
        # The tool output mentions "error" but status is success. Must pass.
        rec = _rec_with(llm_content="done", tool_status="success")
        rec.interactions[0].tool_output = "No errors found in codebase"
        assertions = ReplayAssertions(rec, rec)
        assertions.assert_no_errors()

    def test_errors_detected_from_status(self) -> None:
        rec = _rec_with(llm_content="done", tool_status="error")
        assertions = ReplayAssertions(rec, rec)
        with pytest.raises(AssertionError, match="tool error"):
            assertions.assert_no_errors()


class TestAssertOutputSimilarity:
    def test_exact_match(self) -> None:
        rec = _rec_with(llm_content="The answer is 42.")
        ReplayAssertions(rec, rec).assert_output_similarity()

    def test_wildly_different_fails(self) -> None:
        orig = _rec_with(llm_content="The answer is 42 and all roads lead to Rome.")
        replay = _rec_with(llm_content="Zebra stripes are unique to each animal.")
        with pytest.raises(AssertionError, match="similarity"):
            ReplayAssertions(orig, replay).assert_output_similarity(min_ratio=0.8)

    def test_both_empty_passes(self) -> None:
        orig = _rec_with(llm_content="")
        replay = _rec_with(llm_content="")
        ReplayAssertions(orig, replay).assert_output_similarity()


# ---------------------------------------------------------------------------
# ContinuationTracker sliding-window DR detection
# ---------------------------------------------------------------------------


class TestSlidingWindowDR:
    def test_decline_detected_at_window_boundary(self) -> None:
        """When recent window drops vs the preceding window, DR triggers."""
        tracker = ContinuationTracker(
            budget_tokens=10_000_000,
            max_continuations=100,
            diminishing_returns_ratio=0.5,
            min_turns_for_dr=4,
        )
        # Previous window [5000, 5000]; recent window [500, 500] → ratio 0.1 → DR.
        tracker.record_turn(5000)
        tracker.record_turn(5000)
        tracker.record_turn(500)
        tracker.record_turn(500)
        decision = tracker.should_continue()
        assert decision.diminishing_returns is True
        assert decision.action == "stop"

    def test_single_large_early_turn_does_not_mask_dr(self) -> None:
        """A single massive early turn used to dominate the `earlier` average."""
        tracker = ContinuationTracker(
            budget_tokens=10_000_000,
            max_continuations=100,
            diminishing_returns_ratio=0.3,
            min_turns_for_dr=4,
        )
        # With old algorithm, [50000, 100, 100, 100, 100, 100]: earlier avg
        # (50000+100+100+100)/4 = 12575, recent avg 100 => ratio 0.008 => detected.
        # With windowed: earlier window=[100, 100], recent=[100, 100] => ratio ~1 => no DR.
        # This is the *desired* behavior — gradual stability is not decline.
        tracker.record_turn(50_000)
        tracker.record_turn(100)
        tracker.record_turn(100)
        tracker.record_turn(100)
        tracker.record_turn(100)
        tracker.record_turn(100)
        decision = tracker.should_continue()
        assert decision.diminishing_returns is False

    def test_stable_output_no_dr(self) -> None:
        tracker = ContinuationTracker(
            budget_tokens=10_000_000,
            max_continuations=100,
            diminishing_returns_ratio=0.3,
            min_turns_for_dr=3,
        )
        for _ in range(6):
            tracker.record_turn(1000)
        decision = tracker.should_continue()
        assert decision.diminishing_returns is False
