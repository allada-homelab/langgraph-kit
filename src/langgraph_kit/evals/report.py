"""Report generation — JSON file output and console summary."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from langgraph_kit.evals.models import EvalReport
from langgraph_kit.evals.runner import DEFAULT_PASS_THRESHOLD

logger = logging.getLogger(__name__)

# Bumped when the ci_json schema changes incompatibly. Consumers (CI
# pipelines, baselines stored in artifacts) should refuse to compare
# different major versions.
CI_JSON_SCHEMA_VERSION: int = 1


def write_json_report(report: EvalReport, path: str | Path) -> None:
    """Write the evaluation report as a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    logger.info("Report written to %s", output)


def compute_overall_pass_rate(report: EvalReport) -> float | None:
    """Mean of per-metric pass rates, restricted to metrics that report one.

    Counts each metric equally regardless of trace count — that's the
    contract `--fail-under` thresholds against. Metrics with no
    ``pass_rate`` (e.g. pure NUMERIC mean-only metrics) are excluded
    from the average. Returns ``None`` if no metric reports a pass
    rate, in which case CI thresholds short-circuit (no number to
    threshold against = no automated failure).
    """
    rates = [s.pass_rate for s in report.metrics.values() if s.pass_rate is not None]
    if not rates:
        return None
    return sum(rates) / len(rates)


def report_to_ci_json(report: EvalReport) -> dict[str, Any]:
    """Render a slim, stable JSON shape for CI consumers + baselines.

    Distinct from the full ``model_dump()`` written by
    :func:`write_json_report`: the CI shape is intentionally narrow so
    a stored baseline can survive minor changes to the full report
    without breaking regression checks. Carries
    :data:`CI_JSON_SCHEMA_VERSION` so consumers can refuse to compare
    incompatible versions.
    """
    return {
        "schema_version": CI_JSON_SCHEMA_VERSION,
        "timestamp": report.timestamp,
        "total_traces": report.total_traces,
        "pass_rate": compute_overall_pass_rate(report),
        "metrics": {
            name: {
                "pass_rate": summary.pass_rate,
                "mean": summary.mean,
                "count": summary.count,
            }
            for name, summary in report.metrics.items()
        },
    }


def check_ci_thresholds(
    report: EvalReport,
    *,
    fail_under: float | None,
    baseline: dict[str, Any] | None,
    baseline_tolerance: float,
) -> list[str]:
    """Return a list of CI-failure reasons; an empty list means the run passes.

    Two checks, each only active when their argument is supplied:

    - **`fail_under`**: overall ``pass_rate`` must be at least this value.
      Skipped (no failure) when the report has no pass-rate-bearing
      metrics — there's nothing to threshold against.
    - **`baseline`**: ``pass_rate`` must not have dropped versus the
      stored baseline by more than ``baseline_tolerance``. Schema
      version of the baseline is checked first; mismatch is itself a
      failure so a stale baseline can't silently pass.
    """
    failures: list[str] = []
    overall = compute_overall_pass_rate(report)

    if fail_under is not None and overall is not None and overall < fail_under:
        failures.append(f"pass_rate {overall:.3f} < --fail-under {fail_under:.3f}")

    if baseline is not None:
        base_version = baseline.get("schema_version")
        if base_version != CI_JSON_SCHEMA_VERSION:
            failures.append(
                f"baseline schema_version {base_version!r} != current "
                f"{CI_JSON_SCHEMA_VERSION!r}; refusing to compare"
            )
        else:
            base_rate = baseline.get("pass_rate")
            if (
                isinstance(base_rate, (int, float))
                and overall is not None
                and overall + baseline_tolerance < base_rate
            ):
                failures.append(
                    f"pass_rate dropped from {base_rate:.3f} to {overall:.3f}"
                    f" (tolerance {baseline_tolerance:.3f})"
                )

    return failures


def _out(text: str) -> None:
    """Write a line to stdout (wrapper to satisfy ruff T201)."""
    sys.stdout.write(text + "\n")


def print_console_report(
    report: EvalReport,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    warn_threshold: float = 0.5,
) -> None:
    """Print a human-readable summary to stdout."""
    _out(f"\n{'=' * 60}")
    _out(f"  Evaluation Report — {report.timestamp}")
    _out(f"{'=' * 60}")
    _out(f"  Traces evaluated: {report.total_traces}")
    _out(f"  Duration: {report.duration_seconds}s")
    _out("")

    for name, summary in report.metrics.items():
        status = ""
        if summary.pass_rate is not None:
            if summary.pass_rate >= pass_threshold:
                status = "PASS"
            elif summary.pass_rate >= warn_threshold:
                status = "WARN"
            else:
                status = "FAIL"

        mean_str = f"  mean={summary.mean:.3f}" if summary.mean is not None else ""
        pass_str = (
            f"  pass_rate={summary.pass_rate:.1%}"
            if summary.pass_rate is not None
            else ""
        )
        _out(f"  [{status:4s}] {name} ({summary.count} scored){mean_str}{pass_str}")

    _out(f"\n{'=' * 60}\n")
