"""Data models for token budget tracking."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


def estimate_cost(usage: TokenUsage) -> float:
    """Estimate cost in USD for a token usage record."""
    model = usage.model.lower()
    # Try exact match, then prefix match
    rates = COST_PER_MILLION.get(model)
    if rates is None:
        for key, val in COST_PER_MILLION.items():
            if model.startswith(key):
                rates = val
                break
    if rates is None:
        return 0.0

    input_rate, output_rate = rates
    return (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000


class BudgetSummary(BaseModel):
    """Summary emitted as an SSE event at the end of a stream."""

    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    budget_consumed_pct: float = 0.0
    turns: list[TokenUsage] = Field(default_factory=list)
