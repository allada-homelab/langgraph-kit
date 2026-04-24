"""Context pressure monitoring and mitigation for long-running agent sessions."""

from __future__ import annotations

import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _heuristic_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars/token average for English text.

    Good enough for pressure heuristics. Not accurate enough for billing —
    plug in a real tokenizer via ``PressureMonitor(token_estimator=...)``
    when precise accounting matters.
    """
    return len(text) // 4


class MitigationStrategy(StrEnum):
    """Available strategies for reducing context pressure."""

    NONE = "none"  # No action needed
    MICROCOMPACT = "microcompact"  # Clear old tool outputs
    PARTIAL_COMPACTION = "partial_compaction"  # Summarize older head, keep recent tail
    FULL_COMPACTION = "full_compaction"  # Summarize entire conversation
    STOP = "stop"  # Stop continuation entirely


class PressureSignals(BaseModel):
    """Observed context pressure metrics."""

    estimated_tokens: int
    window_limit: int
    pressure_pct: float
    large_tool_outputs: int
    compaction_failures: int


class PressureMonitor:
    """Monitors context growth and selects appropriate mitigation strategies.

    The monitor maintains a circuit breaker for compaction failures to avoid
    futile retry loops. The breaker opens after ``max_compaction_failures``
    consecutive failures and auto-resets after ``compaction_cooldown_seconds``
    have elapsed — so a long-running session that hits early failures can
    recover instead of being stuck at STOP until the process restarts.
    """

    def __init__(
        self,
        window_limit: int = 128_000,
        warn_pct: float = 0.70,
        critical_pct: float = 0.85,
        max_compaction_failures: int = 3,
        large_output_threshold: int = 4000,
        *,
        compaction_cooldown_seconds: float = 300.0,
        enable_partial_compaction: bool = False,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        super().__init__()
        self._window_limit = window_limit
        self._warn_pct = warn_pct
        self._critical_pct = critical_pct
        self._max_failures = max_compaction_failures
        self._large_threshold = large_output_threshold
        self._cooldown_seconds = compaction_cooldown_seconds
        self._enable_partial = enable_partial_compaction
        self._estimate_tokens: Callable[[str], int] = (
            token_estimator or _heuristic_token_count
        )
        self._compaction_failures: int = 0
        self._breaker_opened_at: float | None = None

    def assess(self, messages: list[Any]) -> PressureSignals:
        """Assess current context pressure from message list."""
        total_tokens = 0
        large_outputs = 0

        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                tokens = self._estimate_tokens(content)
                total_tokens += tokens
                if tokens > self._large_threshold:
                    large_outputs += 1
            elif isinstance(content, list):
                # Multi-part messages — also track large parts
                for part in content:  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(part, dict):
                        text_val: str = str(part.get("text", ""))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        part_tokens = self._estimate_tokens(text_val)
                        total_tokens += part_tokens
                        if part_tokens > self._large_threshold:
                            large_outputs += 1

        pct = total_tokens / self._window_limit if self._window_limit > 0 else 1.0

        return PressureSignals(
            estimated_tokens=total_tokens,
            window_limit=self._window_limit,
            pressure_pct=min(pct, 1.0),
            large_tool_outputs=large_outputs,
            compaction_failures=self._compaction_failures,
        )

    def choose_mitigation(self, signals: PressureSignals) -> MitigationStrategy:
        """Choose the most appropriate mitigation strategy given current pressure."""
        # Circuit breaker: if compaction keeps failing, stop — but auto-recover
        # after the cooldown so we don't strand a long-running session.
        if self._breaker_is_open():
            logger.warning(
                "Compaction circuit breaker open (%d failures)",
                self._compaction_failures,
            )
            return MitigationStrategy.STOP

        # No pressure
        if signals.pressure_pct < self._warn_pct:
            return MitigationStrategy.NONE

        # Moderate pressure: try clearing large tool outputs first
        if signals.pressure_pct < self._critical_pct:
            if signals.large_tool_outputs > 2:
                return MitigationStrategy.MICROCOMPACT
            if self._enable_partial:
                # No large outputs to trim but pressure is rising — summarize
                # the older head while keeping the recent tail intact. Cheaper
                # than waiting for critical pressure and doing a full compaction.
                return MitigationStrategy.PARTIAL_COMPACTION
            return MitigationStrategy.NONE

        # Critical pressure — escalation strategy:
        # If there are large tool outputs, try the cheap/targeted MICROCOMPACT first.
        # It runs without an LLM call and often frees enough space. If it doesn't,
        # the next turn re-enters here with fewer large outputs and falls through
        # to the more expensive FULL_COMPACTION.
        if signals.large_tool_outputs > 3:
            return MitigationStrategy.MICROCOMPACT

        return MitigationStrategy.FULL_COMPACTION

    def record_compaction_failure(self) -> None:
        """Record a failed compaction attempt."""
        self._compaction_failures += 1
        if (
            self._compaction_failures >= self._max_failures
            and self._breaker_opened_at is None
        ):
            self._breaker_opened_at = time.monotonic()

    def record_compaction_success(self) -> None:
        """Reset failure counter after successful compaction."""
        self._compaction_failures = 0
        self._breaker_opened_at = None

    def reset(self) -> None:
        """Reset monitor state."""
        self._compaction_failures = 0
        self._breaker_opened_at = None

    def _breaker_is_open(self) -> bool:
        """Return True if the circuit breaker is currently blocking attempts.

        Auto-resets the failure counter once the cooldown has elapsed so a
        long-running session can recover from an early burst of failures
        (previously the breaker stayed open until process restart).
        """
        if self._compaction_failures < self._max_failures:
            return False
        if self._breaker_opened_at is None:
            # Defensive — should have been set in record_compaction_failure,
            # but cope gracefully if callers bypass it.
            self._breaker_opened_at = time.monotonic()
            return True
        if self._cooldown_seconds <= 0:
            return True  # cooldown disabled — breaker stays open until success
        elapsed = time.monotonic() - self._breaker_opened_at
        if elapsed >= self._cooldown_seconds:
            logger.info(
                "Compaction circuit breaker cooldown elapsed (%.0fs) — resetting",
                elapsed,
            )
            self._compaction_failures = 0
            self._breaker_opened_at = None
            return False
        return True
