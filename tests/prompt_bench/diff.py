"""Diff and report generation for two bench runs.

Pairs base/variant samples by ``scenario_id + sample_index``, runs the
pairwise judge panel on each pair, computes per-scenario and overall
win rates / agreement, applies rule-based metrics, and renders a
markdown report fit for an issue comment or PR artifact.

Strict acceptance criteria (the bar for shipping a prompt change):

1. Pairwise win rate >= 60% across decided pairs.
2. Two-judge agreement >= 70%.
3. No rule-based metric regression > 5%.
4. (Caller's responsibility) cross-prompt regression suite passes.
5. (Caller's responsibility) high-variance scenarios re-run with N=10.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph_kit.evals.metrics.rule_based import (
    ErrorFreeMetric,
    LatencyMetric,
    ResponseLengthMetric,
    ToolEfficiencyMetric,
)

if TYPE_CHECKING:
    from langgraph_kit.evals.models import EvalMetric
    from tests.prompt_bench.pairwise import PairwisePanel
    from tests.prompt_bench.runner import BenchReport, BenchSample


logger = logging.getLogger(__name__)


# Acceptance thresholds — the strict bar we agreed to in the plan.
WIN_RATE_THRESHOLD = 0.60
JUDGE_AGREEMENT_THRESHOLD = 0.70
METRIC_REGRESSION_TOLERANCE = 0.05


def default_rule_metrics() -> list[EvalMetric]:
    """The four rule-based metrics applied to every diff."""
    return [
        LatencyMetric(),
        ErrorFreeMetric(),
        ToolEfficiencyMetric(),
        ResponseLengthMetric(),
    ]


@dataclass
class ScenarioDiff:
    """Diff result for a single scenario."""

    scenario_id: str
    pair_count: int = 0
    decided_pairs: int = 0
    base_wins: int = 0
    variant_wins: int = 0
    ties: int = 0
    judge_agreement: float = 0.0
    base_metric_means: dict[str, float] = field(default_factory=dict)
    variant_metric_means: dict[str, float] = field(default_factory=dict)
    sample_score_iqr: float = 0.0

    @property
    def decided_win_rate(self) -> float:
        if self.decided_pairs == 0:
            return 0.0
        return round(self.variant_wins / self.decided_pairs, 3)

    @property
    def metric_deltas(self) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for name, base_mean in self.base_metric_means.items():
            var_mean = self.variant_metric_means.get(name, base_mean)
            deltas[name] = round(var_mean - base_mean, 3)
        return deltas


@dataclass
class DiffReport:
    """Aggregate diff across all scenarios."""

    base_overlay: str
    variant_overlay: str
    scenarios: list[ScenarioDiff] = field(default_factory=list)

    @property
    def total_pairs(self) -> int:
        return sum(s.pair_count for s in self.scenarios)

    @property
    def total_decided(self) -> int:
        return sum(s.decided_pairs for s in self.scenarios)

    @property
    def total_variant_wins(self) -> int:
        return sum(s.variant_wins for s in self.scenarios)

    @property
    def total_base_wins(self) -> int:
        return sum(s.base_wins for s in self.scenarios)

    @property
    def total_ties(self) -> int:
        return sum(s.ties for s in self.scenarios)

    @property
    def overall_win_rate(self) -> float:
        if self.total_decided == 0:
            return 0.0
        return round(self.total_variant_wins / self.total_decided, 3)

    @property
    def overall_judge_agreement(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return round(self.total_decided / self.total_pairs, 3)

    @property
    def regression_flags(self) -> list[str]:
        """List of metric names that regressed beyond tolerance."""
        flags: list[str] = []
        for scenario in self.scenarios:
            for name, delta in scenario.metric_deltas.items():
                if delta < -METRIC_REGRESSION_TOLERANCE and name not in flags:
                    flags.append(name)
        return flags

    @property
    def passes_acceptance(self) -> bool:
        return (
            self.overall_win_rate >= WIN_RATE_THRESHOLD
            and self.overall_judge_agreement >= JUDGE_AGREEMENT_THRESHOLD
            and not self.regression_flags
        )


async def compute_diff(
    base_report: BenchReport,
    variant_report: BenchReport,
    panel: PairwisePanel,
    rule_metrics: list[EvalMetric] | None = None,
) -> DiffReport:
    """Diff base vs variant — pairwise + rule-based metric deltas."""
    metrics = rule_metrics or default_rule_metrics()
    diff = DiffReport(
        base_overlay=base_report.overlay_name,
        variant_overlay=variant_report.overlay_name,
    )

    by_scenario_base = _group_by_scenario(base_report.samples)
    by_scenario_variant = _group_by_scenario(variant_report.samples)

    for scenario_id, base_samples in by_scenario_base.items():
        variant_samples = by_scenario_variant.get(scenario_id, [])
        scenario_diff = await _diff_scenario(
            scenario_id, base_samples, variant_samples, panel, metrics
        )
        diff.scenarios.append(scenario_diff)

    diff.scenarios.sort(key=lambda s: s.scenario_id)
    return diff


async def _diff_scenario(
    scenario_id: str,
    base_samples: list[BenchSample],
    variant_samples: list[BenchSample],
    panel: PairwisePanel,
    metrics: list[EvalMetric],
) -> ScenarioDiff:
    diff = ScenarioDiff(scenario_id=scenario_id)
    pairs = list(zip(base_samples, variant_samples, strict=False))
    diff.pair_count = len(pairs)

    for base_sample, variant_sample in pairs:
        result = await panel.compare(
            input_text=scenario_id,
            base_trace=base_sample.to_trace(),
            variant_trace=variant_sample.to_trace(),
        )
        if result.decided:
            diff.decided_pairs += 1
            if result.winner == "tie":
                diff.ties += 1
            elif result.base_won:
                diff.base_wins += 1
            else:
                diff.variant_wins += 1

    diff.judge_agreement = (
        round(diff.decided_pairs / diff.pair_count, 3) if diff.pair_count else 0.0
    )

    diff.base_metric_means = await _metric_means(metrics, base_samples)
    diff.variant_metric_means = await _metric_means(metrics, variant_samples)

    base_scores = [s.duration_ms for s in base_samples if s.duration_ms]
    if len(base_scores) >= 2:
        try:
            qs = statistics.quantiles(base_scores, n=4)
            diff.sample_score_iqr = round(qs[2] - qs[0], 3)
        except statistics.StatisticsError:
            diff.sample_score_iqr = 0.0

    return diff


async def _metric_means(
    metrics: list[EvalMetric], samples: list[BenchSample]
) -> dict[str, float]:
    means: dict[str, float] = {}
    if not samples:
        return means
    for metric in metrics:
        values: list[float] = []
        for sample in samples:
            try:
                result = await metric.score(sample.to_trace())
            except Exception:
                logger.exception(
                    "Rule metric %s failed on sample %s",
                    metric.name,
                    sample.scenario_id,
                )
                continue
            value = result.value
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            means[metric.name] = round(sum(values) / len(values), 3)
    return means


def _group_by_scenario(samples: list[BenchSample]) -> dict[str, list[BenchSample]]:
    grouped: dict[str, list[BenchSample]] = {}
    for s in samples:
        grouped.setdefault(s.scenario_id, []).append(s)
    for v in grouped.values():
        v.sort(key=lambda s: s.sample_index)
    return grouped


def render_markdown(diff: DiffReport) -> str:
    """Render the diff as a markdown report (issue-comment friendly)."""
    lines: list[str] = []
    pass_str = "PASS" if diff.passes_acceptance else "FAIL"
    lines.append(
        f"# Prompt-bench diff: `{diff.variant_overlay}` vs `{diff.base_overlay}` — **{pass_str}**"
    )
    lines.append("")
    lines.append("## Acceptance criteria")
    lines.append("")
    win_status = "meets" if diff.overall_win_rate >= WIN_RATE_THRESHOLD else "BELOW"
    lines.append(
        f"- Win rate: **{diff.overall_win_rate:.1%}** "
        + f"({win_status} {WIN_RATE_THRESHOLD:.0%} bar)"
    )
    agree_status = (
        "meets"
        if diff.overall_judge_agreement >= JUDGE_AGREEMENT_THRESHOLD
        else "BELOW"
    )
    lines.append(
        f"- Judge agreement: **{diff.overall_judge_agreement:.1%}** "
        + f"({agree_status} {JUDGE_AGREEMENT_THRESHOLD:.0%} bar)"
    )
    if diff.regression_flags:
        lines.append(f"- Regressions flagged on: {', '.join(diff.regression_flags)}")
    else:
        lines.append("- No rule-based metric regressions beyond tolerance")
    lines.append("")
    lines.append(
        f"## Totals — {diff.total_decided}/{diff.total_pairs} decided "
        + f"({diff.total_variant_wins} variant wins, "
        + f"{diff.total_base_wins} base wins, {diff.total_ties} ties)"
    )
    lines.append("")
    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append(
        "| Scenario | Decided | Variant wins | Base wins | Ties | Win rate | Metric deltas |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in diff.scenarios:
        deltas_str = (
            ", ".join(f"{name}={delta:+.3f}" for name, delta in s.metric_deltas.items())
            or "-"
        )
        row = (
            f"| `{s.scenario_id}` | {s.decided_pairs}/{s.pair_count} | "
            + f"{s.variant_wins} | {s.base_wins} | {s.ties} | "
            + f"{s.decided_win_rate:.1%} | {deltas_str} |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def to_dict(diff: DiffReport) -> dict[str, Any]:
    """Serializable dict for JSON artifact."""
    return {
        "base_overlay": diff.base_overlay,
        "variant_overlay": diff.variant_overlay,
        "totals": {
            "pair_count": diff.total_pairs,
            "decided": diff.total_decided,
            "variant_wins": diff.total_variant_wins,
            "base_wins": diff.total_base_wins,
            "ties": diff.total_ties,
            "win_rate": diff.overall_win_rate,
            "judge_agreement": diff.overall_judge_agreement,
        },
        "regression_flags": diff.regression_flags,
        "passes_acceptance": diff.passes_acceptance,
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "pair_count": s.pair_count,
                "decided_pairs": s.decided_pairs,
                "base_wins": s.base_wins,
                "variant_wins": s.variant_wins,
                "ties": s.ties,
                "judge_agreement": s.judge_agreement,
                "decided_win_rate": s.decided_win_rate,
                "base_metric_means": s.base_metric_means,
                "variant_metric_means": s.variant_metric_means,
                "metric_deltas": s.metric_deltas,
                "sample_score_iqr": s.sample_score_iqr,
            }
            for s in diff.scenarios
        ],
    }
