"""Coverage — CI threshold + baseline + ci-json helpers added by issue #14.

The `python -m langgraph_kit.evals` CLI now exits non-zero when:

- the overall pass rate (mean across pass-rate-bearing metrics) is
  below ``--fail-under``
- pass rate has dropped below the stored ``--baseline`` by more than
  ``--baseline-tolerance``
- the baseline file's ``schema_version`` doesn't match the current
  format

The slim ``--ci-json`` output is intentionally separate from the full
``write_json_report`` output: a future change to the full report's
shape shouldn't invalidate stored baselines.
"""

from __future__ import annotations

from langgraph_kit.evals.models import EvalReport, MetricSummary
from langgraph_kit.evals.report import (
    CI_JSON_SCHEMA_VERSION,
    check_ci_thresholds,
    compute_overall_pass_rate,
    report_to_ci_json,
)


def _report(*pass_rates: float | None) -> EvalReport:
    """Build a minimal EvalReport with N metrics, each carrying a fixed pass_rate."""
    metrics: dict[str, MetricSummary] = {}
    for i, rate in enumerate(pass_rates):
        metrics[f"metric_{i}"] = MetricSummary(
            name=f"metric_{i}",
            data_type="BOOLEAN",
            count=10,
            mean=rate,
            pass_rate=rate,
        )
    return EvalReport(
        timestamp="2026-04-25T00:00:00Z",
        total_traces=10,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# compute_overall_pass_rate
# ---------------------------------------------------------------------------


def test_overall_pass_rate_is_mean_of_metric_pass_rates() -> None:
    report = _report(0.8, 0.6, 1.0)
    overall = compute_overall_pass_rate(report)
    assert overall is not None
    assert abs(overall - 0.8) < 1e-9


def test_overall_pass_rate_skips_metrics_without_a_pass_rate() -> None:
    """Metrics that don't report pass_rate (NUMERIC means) shouldn't drag
    the overall down, nor be counted as 0."""
    report = _report(0.8, None, 1.0)
    overall = compute_overall_pass_rate(report)
    assert overall is not None
    assert abs(overall - 0.9) < 1e-9


def test_overall_pass_rate_returns_none_when_no_metric_reports() -> None:
    """No pass_rate-bearing metrics → no overall to threshold against."""
    report = _report(None, None)
    assert compute_overall_pass_rate(report) is None


# ---------------------------------------------------------------------------
# report_to_ci_json
# ---------------------------------------------------------------------------


def test_ci_json_carries_schema_version_and_overall_pass_rate() -> None:
    payload = report_to_ci_json(_report(0.8, 1.0))
    assert payload["schema_version"] == CI_JSON_SCHEMA_VERSION
    assert payload["pass_rate"] is not None
    assert "metric_0" in payload["metrics"]
    assert payload["metrics"]["metric_0"]["pass_rate"] == 0.8


def test_ci_json_handles_empty_metrics() -> None:
    payload = report_to_ci_json(_report())
    assert payload["pass_rate"] is None
    assert payload["metrics"] == {}


# ---------------------------------------------------------------------------
# check_ci_thresholds
# ---------------------------------------------------------------------------


def test_passes_under_threshold_returns_failure() -> None:
    failures = check_ci_thresholds(
        _report(0.5),
        fail_under=0.7,
        baseline=None,
        baseline_tolerance=0.0,
    )
    assert len(failures) == 1
    assert "fail-under" in failures[0].lower()


def test_at_threshold_passes() -> None:
    """``--fail-under 0.7`` against a 0.7 pass_rate should not fail
    (strict less-than, not less-than-or-equal)."""
    failures = check_ci_thresholds(
        _report(0.7),
        fail_under=0.7,
        baseline=None,
        baseline_tolerance=0.0,
    )
    assert failures == []


def test_no_metrics_with_pass_rate_skips_fail_under() -> None:
    """When there's no overall pass_rate, --fail-under has nothing to
    threshold against → don't fail the build."""
    failures = check_ci_thresholds(
        _report(None, None),
        fail_under=0.99,
        baseline=None,
        baseline_tolerance=0.0,
    )
    assert failures == []


def test_baseline_regression_fails() -> None:
    failures = check_ci_thresholds(
        _report(0.6),
        fail_under=None,
        baseline={"schema_version": CI_JSON_SCHEMA_VERSION, "pass_rate": 0.9},
        baseline_tolerance=0.0,
    )
    assert len(failures) == 1
    assert "dropped" in failures[0].lower()


def test_baseline_within_tolerance_passes() -> None:
    failures = check_ci_thresholds(
        _report(0.85),
        fail_under=None,
        baseline={"schema_version": CI_JSON_SCHEMA_VERSION, "pass_rate": 0.9},
        baseline_tolerance=0.1,
    )
    assert failures == []


def test_baseline_improvement_passes() -> None:
    failures = check_ci_thresholds(
        _report(0.95),
        fail_under=None,
        baseline={"schema_version": CI_JSON_SCHEMA_VERSION, "pass_rate": 0.9},
        baseline_tolerance=0.0,
    )
    assert failures == []


def test_baseline_schema_mismatch_fails() -> None:
    """A baseline from an older schema version should be refused
    rather than silently compared — otherwise a CI pipeline can stop
    catching regressions when the schema bumps."""
    failures = check_ci_thresholds(
        _report(0.5),
        fail_under=None,
        baseline={"schema_version": 0, "pass_rate": 0.9},
        baseline_tolerance=0.0,
    )
    assert len(failures) == 1
    assert "schema" in failures[0].lower()


def test_both_checks_can_fail_simultaneously() -> None:
    failures = check_ci_thresholds(
        _report(0.4),
        fail_under=0.8,
        baseline={"schema_version": CI_JSON_SCHEMA_VERSION, "pass_rate": 0.9},
        baseline_tolerance=0.0,
    )
    assert len(failures) == 2


def test_no_args_means_no_failures() -> None:
    """Without --fail-under and --baseline, the CI checks are no-ops."""
    failures = check_ci_thresholds(
        _report(0.0),
        fail_under=None,
        baseline=None,
        baseline_tolerance=0.0,
    )
    assert failures == []
