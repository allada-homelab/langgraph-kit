"""Tests for the diff aggregator and acceptance-criteria logic."""

from __future__ import annotations

import json
import random
from typing import Any

from tests.prompt_bench.diff import (
    JUDGE_AGREEMENT_THRESHOLD,
    METRIC_REGRESSION_TOLERANCE,
    WIN_RATE_THRESHOLD,
    DiffReport,
    ScenarioDiff,
    compute_diff,
    render_markdown,
    to_dict,
)
from tests.prompt_bench.pairwise import PairwiseJudge, PairwisePanel
from tests.prompt_bench.runner import BenchReport, BenchSample


class _CannedJudge:
    """Always returns the same decision JSON. Lets us drive diff aggregation."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__()
        self._payload = json.dumps(response)

    async def ainvoke(self, _messages: list[Any]) -> Any:
        class _R:
            content = self._payload

        return _R()


def _panel(winner: str) -> PairwisePanel:
    judges = [
        PairwiseJudge(
            name=f"j{i}",
            llm=_CannedJudge({"winner": winner, "confidence": 0.9, "reason": "x"}),
            rubric_text="dummy",
        )
        for i in range(2)
    ]
    return PairwisePanel(judges=judges, rng=random.Random(0))


def _bench_report(name: str, sample_count: int = 5) -> BenchReport:
    samples = [
        BenchSample(
            scenario_id="s",
            sample_index=i,
            overlay_name=name,
            duration_ms=1000.0,
            final_output=f"output {name} {i}",
        )
        for i in range(sample_count)
    ]
    return BenchReport(overlay_name=name, samples=samples)


class TestComputeDiff:
    async def test_consensus_b_with_seeded_rng(self) -> None:
        # rng seed=0 → first call's choice is True → base_was_a=True
        # so "B" winner means variant won. 5/5 variant wins → 100% win rate.
        diff = await compute_diff(
            base_report=_bench_report("base"),
            variant_report=_bench_report("variant"),
            panel=_panel("B"),
        )
        assert diff.total_pairs == 5
        # With a single seeded RNG the panel's choice is deterministic
        # *given the order of calls*. We don't assert on exact wins
        # because randomization may flip A/B per call; instead we
        # assert decisive consensus and full-decided count.
        assert diff.total_decided == 5
        assert diff.total_variant_wins + diff.total_base_wins == 5
        assert diff.total_ties == 0

    async def test_unanimous_tie_is_decided(self) -> None:
        diff = await compute_diff(
            base_report=_bench_report("base"),
            variant_report=_bench_report("variant"),
            panel=_panel("tie"),
        )
        assert diff.total_decided == 5
        assert diff.total_ties == 5
        assert diff.overall_win_rate == 0.0  # variant didn't win any


class TestAcceptanceCriteria:
    def test_passes_when_all_thresholds_met(self) -> None:
        diff = DiffReport(
            base_overlay="base",
            variant_overlay="variant",
            scenarios=[
                ScenarioDiff(
                    scenario_id="s1",
                    pair_count=10,
                    decided_pairs=10,
                    base_wins=2,
                    variant_wins=7,
                    ties=1,
                    judge_agreement=1.0,
                )
            ],
        )
        # 7/10 decided → 70% win rate, 100% agreement, no regressions
        assert diff.overall_win_rate >= WIN_RATE_THRESHOLD
        assert diff.overall_judge_agreement >= JUDGE_AGREEMENT_THRESHOLD
        assert diff.passes_acceptance

    def test_fails_below_win_rate(self) -> None:
        diff = DiffReport(
            base_overlay="base",
            variant_overlay="variant",
            scenarios=[
                ScenarioDiff(
                    scenario_id="s1",
                    pair_count=10,
                    decided_pairs=10,
                    base_wins=5,
                    variant_wins=5,
                    ties=0,
                    judge_agreement=1.0,
                )
            ],
        )
        # 5/10 = 50% win rate, below 60% bar
        assert not diff.passes_acceptance

    def test_fails_below_agreement(self) -> None:
        diff = DiffReport(
            base_overlay="base",
            variant_overlay="variant",
            scenarios=[
                ScenarioDiff(
                    scenario_id="s1",
                    pair_count=10,
                    decided_pairs=5,
                    base_wins=1,
                    variant_wins=4,
                    ties=0,
                    judge_agreement=0.5,
                )
            ],
        )
        # 4/5 = 80% win rate but 5/10 = 50% agreement, below 70% bar
        assert diff.overall_win_rate >= WIN_RATE_THRESHOLD
        assert diff.overall_judge_agreement < JUDGE_AGREEMENT_THRESHOLD
        assert not diff.passes_acceptance

    def test_flags_regressions(self) -> None:
        diff = DiffReport(
            base_overlay="base",
            variant_overlay="variant",
            scenarios=[
                ScenarioDiff(
                    scenario_id="s1",
                    pair_count=10,
                    decided_pairs=10,
                    base_wins=2,
                    variant_wins=8,
                    ties=0,
                    judge_agreement=1.0,
                    base_metric_means={"latency": 0.95, "error_free": 1.0},
                    variant_metric_means={
                        "latency": 0.95 - METRIC_REGRESSION_TOLERANCE - 0.01,
                        "error_free": 1.0,
                    },
                )
            ],
        )
        assert "latency" in diff.regression_flags
        assert "error_free" not in diff.regression_flags
        assert not diff.passes_acceptance


class TestRendering:
    def test_markdown_renders_pass(self) -> None:
        diff = DiffReport(
            base_overlay="base",
            variant_overlay="variant",
            scenarios=[
                ScenarioDiff(
                    scenario_id="s1",
                    pair_count=10,
                    decided_pairs=10,
                    variant_wins=7,
                    base_wins=2,
                    ties=1,
                    judge_agreement=1.0,
                )
            ],
        )
        md = render_markdown(diff)
        assert "PASS" in md
        assert "70.0%" in md or "70%" in md
        assert "s1" in md

    def test_to_dict_round_trip(self) -> None:
        diff = DiffReport(
            base_overlay="b",
            variant_overlay="v",
            scenarios=[ScenarioDiff(scenario_id="s")],
        )
        payload = to_dict(diff)
        # Should be JSON-serializable
        json.dumps(payload)


class TestScenarioDiffWinRate:
    def test_zero_when_no_decisions(self) -> None:
        sd = ScenarioDiff(scenario_id="s", pair_count=5, decided_pairs=0)
        assert sd.decided_win_rate == 0.0
