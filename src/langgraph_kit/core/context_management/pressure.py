"""Context pressure monitoring and mitigation for long-running agent sessions."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MitigationStrategy(str, Enum):
    """Available strategies for reducing context pressure."""

    NONE = "none"  # No action needed
    MICROCOMPACT = "microcompact"  # Clear old tool outputs
    SESSION_ASSISTED = "session_assisted"  # Use notebook as continuity anchor
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

    The monitor maintains a simple circuit breaker for compaction failures
    to avoid futile retry loops.
    """

    def __init__(
        self,
        window_limit: int = 128_000,
        warn_pct: float = 0.70,
        critical_pct: float = 0.85,
        max_compaction_failures: int = 3,
        large_output_threshold: int = 4000,
    ) -> None:
        self._window_limit = window_limit
        self._warn_pct = warn_pct
        self._critical_pct = critical_pct
        self._max_failures = max_compaction_failures
        self._large_threshold = large_output_threshold
        self._compaction_failures: int = 0

    def assess(self, messages: list[Any]) -> PressureSignals:
        """Assess current context pressure from message list."""
        total_tokens = 0
        large_outputs = 0

        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                # ~4 chars per token is a rough average for English text.
                # Good enough for pressure heuristics; not used for billing.
                tokens = len(content) // 4
                total_tokens += tokens
                if tokens > self._large_threshold:
                    large_outputs += 1
            elif isinstance(content, list):
                # Multi-part messages
                for part in content:  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(part, dict):
                        text_val: str = str(part.get("text", ""))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        part_tokens = len(text_val) // 4
                        total_tokens += part_tokens

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
        # Circuit breaker: if compaction keeps failing, stop
        if self._compaction_failures >= self._max_failures:
            logger.warning(
                "Compaction circuit breaker triggered (%d failures)",
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

    def record_compaction_success(self) -> None:
        """Reset failure counter after successful compaction."""
        self._compaction_failures = 0

    def reset(self) -> None:
        """Reset monitor state."""
        self._compaction_failures = 0
