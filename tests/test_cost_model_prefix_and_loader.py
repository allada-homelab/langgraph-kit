"""Regression: estimate_cost prefix ordering + runtime rate loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langgraph_kit.core.cost.models import (
    COST_PER_MILLION,
    TokenUsage,
    estimate_cost,
    load_rates_from_json,
)


def test_longest_prefix_match_wins_over_shorter() -> None:
    """``claude-sonnet-4-6`` must beat the hypothetical shorter ``claude-sonnet-4``."""
    original = dict(COST_PER_MILLION)
    try:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(
            {
                "claude-sonnet-4": (9.99, 99.99),
                "claude-sonnet-4-6": (3.00, 15.00),
            }
        )

        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
            model="claude-sonnet-4-6-20260601",
        )
        cost = estimate_cost(usage)
        # Sum of input+output rates from the *specific* entry, not the
        # cheaper broad one.
        assert cost == pytest.approx(3.00 + 15.00)
    finally:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(original)


def test_exact_match_preferred_over_prefix() -> None:
    original = dict(COST_PER_MILLION)
    try:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(
            {
                "gpt-4o": (2.50, 10.00),
                "gpt-4o-mini": (0.15, 0.60),
            }
        )

        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            model="gpt-4o",
        )
        cost = estimate_cost(usage)
        assert cost == pytest.approx(2.50)
    finally:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(original)


def test_unknown_model_returns_zero() -> None:
    usage = TokenUsage(
        input_tokens=500_000,
        output_tokens=500_000,
        total_tokens=1_000_000,
        model="some-future-model-never-seen",
    )
    assert estimate_cost(usage) == 0.0


def test_load_rates_replaces_table(tmp_path: Path) -> None:
    original = dict(COST_PER_MILLION)
    try:
        rates_file = tmp_path / "rates.json"
        rates_file.write_text(
            json.dumps({"custom-model": [1.23, 4.56]}), encoding="utf-8"
        )
        load_rates_from_json(rates_file)

        assert "custom-model" in COST_PER_MILLION
        assert "gpt-4o" not in COST_PER_MILLION  # previous entries replaced

        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
            model="custom-model",
        )
        assert estimate_cost(usage) == pytest.approx(1.23 + 4.56)
    finally:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(original)


def test_load_rates_skips_malformed_entries(tmp_path: Path) -> None:
    original = dict(COST_PER_MILLION)
    try:
        rates_file = tmp_path / "rates.json"
        rates_file.write_text(
            json.dumps(
                {
                    "good": [1.0, 2.0],
                    "bad-list-len": [1.0],
                    "bad-not-list": "oops",
                    "bad-str-values": ["1", "2"],
                }
            ),
            encoding="utf-8",
        )
        load_rates_from_json(rates_file)

        assert "good" in COST_PER_MILLION
        assert "bad-list-len" not in COST_PER_MILLION
        assert "bad-not-list" not in COST_PER_MILLION
        assert "bad-str-values" not in COST_PER_MILLION
    finally:
        COST_PER_MILLION.clear()
        COST_PER_MILLION.update(original)
