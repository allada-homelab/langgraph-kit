"""Token-budget continuation policy with diminishing-returns detection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ContinuationDecision(BaseModel):
    """Structured outcome of a continuation policy check."""

    action: Literal["continue", "stop"]
    reason: str
    budget_consumed_pct: float
    continuation_count: int
    total_tokens_used: int
    diminishing_returns: bool


class ContinuationTracker:
    """Tracks per-request continuation state and decides whether to keep going.

    The tracker monitors:
    - How many continuations have occurred
    - How much of the token budget has been consumed
    - Whether recent turns show diminishing returns (token delta shrinking)
    """

    def __init__(
        self,
        budget_tokens: int = 100_000,
        max_continuations: int = 20,
        stop_threshold_pct: float = 0.90,
        diminishing_returns_ratio: float = 0.3,
        min_turns_for_dr: int = 3,
    ) -> None:
        super().__init__()
        self._budget = budget_tokens
        self._max_continuations = max_continuations
        self._stop_threshold = stop_threshold_pct
        self._dr_ratio = diminishing_returns_ratio
        self._min_turns_for_dr = max(min_turns_for_dr, 3)  # needs ≥3 turns for split

        self._continuation_count: int = 0
        self._total_tokens: int = 0
        self._turn_deltas: list[int] = []

    def record_turn(self, tokens_used: int) -> None:
        """Record tokens consumed in the latest turn."""
        self._turn_deltas.append(tokens_used)
        self._total_tokens += tokens_used
        self._continuation_count += 1

    def should_continue(self) -> ContinuationDecision:
        """Evaluate whether the system should continue working."""
        pct = self._total_tokens / self._budget if self._budget > 0 else 1.0
        dr = self._detect_diminishing_returns()

        # Check hard limits
        if self._continuation_count >= self._max_continuations:
            return ContinuationDecision(
                action="stop",
                reason=f"Max continuations reached ({self._max_continuations})",
                budget_consumed_pct=pct,
                continuation_count=self._continuation_count,
                total_tokens_used=self._total_tokens,
                diminishing_returns=dr,
            )

        if pct >= self._stop_threshold:
            return ContinuationDecision(
                action="stop",
                reason=f"Budget nearly exhausted ({pct:.0%} consumed)",
                budget_consumed_pct=pct,
                continuation_count=self._continuation_count,
                total_tokens_used=self._total_tokens,
                diminishing_returns=dr,
            )

        if dr:
            return ContinuationDecision(
                action="stop",
                reason="Diminishing returns detected — recent turns produced less output",
                budget_consumed_pct=pct,
                continuation_count=self._continuation_count,
                total_tokens_used=self._total_tokens,
                diminishing_returns=True,
            )

        return ContinuationDecision(
            action="continue",
            reason="Budget available and progress is productive",
            budget_consumed_pct=pct,
            continuation_count=self._continuation_count,
            total_tokens_used=self._total_tokens,
            diminishing_returns=False,
        )

    def reset(self) -> None:
        """Reset tracker for a new request."""
        self._continuation_count = 0
        self._total_tokens = 0
        self._turn_deltas = []

    def _detect_diminishing_returns(self) -> bool:
        """Check if recent token deltas are shrinking significantly.

        Compares the average of the last two turns against the average of
        earlier turns. If the recent average is below `diminishing_returns_ratio`
        of the earlier average, returns True.
        """
        if len(self._turn_deltas) < self._min_turns_for_dr:
            return False

        recent = self._turn_deltas[-2:]
        earlier = self._turn_deltas[:-2]

        if not earlier:
            return False

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        if earlier_avg <= 0:
            return False

        return (recent_avg / earlier_avg) < self._dr_ratio
