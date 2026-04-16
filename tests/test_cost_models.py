"""Tests for cost models — estimate_cost and TokenUsage."""

from __future__ import annotations

from langgraph_kit.core.cost.models import TokenUsage, estimate_cost


class TestEstimateCost:
    def test_exact_model_match(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, model="gpt-4o")
        cost = estimate_cost(usage)
        assert cost == 2.50  # $2.50 per 1M input tokens

    def test_output_tokens(self) -> None:
        usage = TokenUsage(input_tokens=0, output_tokens=1_000_000, model="gpt-4o")
        cost = estimate_cost(usage)
        assert cost == 10.00  # $10.00 per 1M output tokens

    def test_mixed_usage(self) -> None:
        usage = TokenUsage(
            input_tokens=100_000, output_tokens=50_000, model="claude-sonnet-4-6"
        )
        cost = estimate_cost(usage)
        # (100k * 3.00 + 50k * 15.00) / 1M = (300_000 + 750_000) / 1M = 1.05
        assert abs(cost - 1.05) < 0.001

    def test_prefix_match(self) -> None:
        usage = TokenUsage(
            input_tokens=1_000_000, output_tokens=0, model="gpt-4o-2024-08-06"
        )
        cost = estimate_cost(usage)
        assert cost == 2.50  # matches gpt-4o prefix

    def test_unknown_model_returns_zero(self) -> None:
        usage = TokenUsage(
            input_tokens=100_000, output_tokens=100_000, model="unknown-model"
        )
        cost = estimate_cost(usage)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        usage = TokenUsage(input_tokens=0, output_tokens=0, model="gpt-4o")
        cost = estimate_cost(usage)
        assert cost == 0.0

    def test_anthropic_opus(self) -> None:
        usage = TokenUsage(
            input_tokens=1_000_000, output_tokens=1_000_000, model="claude-opus-4-6"
        )
        cost = estimate_cost(usage)
        assert cost == 90.0  # 15.00 + 75.00

    def test_case_insensitive(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, model="GPT-4o")
        cost = estimate_cost(usage)
        assert cost == 2.50
