"""Data models for token budget tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Token usage from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0


class BudgetConfig(BaseModel):
    """Configuration for token budget limits."""

    max_tokens_per_thread: int = 0  # 0 = unlimited
    warning_threshold_pct: float = 0.80
    downgrade_model: str = ""  # model to switch to when budget is tight


class BudgetState(BaseModel):
    """Persisted state of token consumption for a thread."""

    thread_id: str = ""
    user_id: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class BudgetCheckResult(BaseModel):
    """Result of a budget check before an LLM call."""

    action: Literal["allow", "warn", "downgrade", "deny"] = "allow"
    reason: str = ""
    budget_consumed_pct: float = 0.0
    remaining_tokens: int = 0


# ---------------------------------------------------------------------------
# Cost lookup table (USD per 1M tokens, input/output)
# Prices as of 2026-04. Used for budget estimates, not billing.
# ---------------------------------------------------------------------------

COST_PER_MILLION: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4-turbo": (10.00, 30.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


def load_rates_from_json(path: Path | str) -> None:
    """Replace the in-memory cost table with values loaded from a JSON file.

    Lets consumers ship their own rates without patching the kit. File
    format: ``{"model-name": [input_rate, output_rate], ...}`` where the
    rates are USD per 1M tokens. Unknown keys are tolerated; values that
    are not 2-element numeric lists are skipped with a warning.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Rates file must contain a JSON object: {path!s}")
    replacement: dict[str, tuple[float, float]] = {}
    for key, value in data.items():  # type: ignore[misc]
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2  # pyright: ignore[reportUnknownArgumentType]
            and all(isinstance(v, (int, float)) for v in value)  # pyright: ignore[reportUnknownVariableType]
        ):
            replacement[str(key)] = (float(value[0]), float(value[1]))
        else:
            logger.warning("Skipping malformed cost entry %r -> %r", key, value)
    COST_PER_MILLION.clear()
    COST_PER_MILLION.update(replacement)


def estimate_cost(usage: TokenUsage) -> float:
    """Estimate cost in USD for a token usage record.

    Matches model name against ``COST_PER_MILLION`` entries using:
      1. Exact lowercase match.
      2. Longest-prefix match — sorting keys by length descending so that
         more-specific entries win ("claude-sonnet-4-6" beats
         "claude-sonnet-4", "gpt-4o-mini" beats "gpt-4").

    Returns ``0.0`` when no key matches (the caller can treat that as
    "unknown model" and optionally log).
    """
    model = usage.model.lower()
    # Exact match first.
    rates = COST_PER_MILLION.get(model)
    if rates is None:
        # Longest-prefix win: sort keys by length descending.
        for key in sorted(COST_PER_MILLION.keys(), key=len, reverse=True):
            if model.startswith(key):
                rates = COST_PER_MILLION[key]
                break
    if rates is None:
        return 0.0

    input_rate, output_rate = rates
    return (
        usage.input_tokens * input_rate + usage.output_tokens * output_rate
    ) / 1_000_000


class BudgetSummary(BaseModel):
    """Summary emitted as an SSE event at the end of a stream."""

    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    budget_consumed_pct: float = 0.0
    turns: list[TokenUsage] = Field(default_factory=list)
