"""Report generation — JSON file output and console summary."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from langgraph_kit.evals.models import EvalReport

logger = logging.getLogger(__name__)


def write_json_report(report: EvalReport, path: str | Path) -> None:
    """Write the evaluation report as a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    logger.info("Report written to %s", output)


def _out(text: str) -> None:
    """Write a line to stdout (wrapper to satisfy ruff T201)."""
    sys.stdout.write(text + "\n")


def print_console_report(
    report: EvalReport,
    *,
    pass_threshold: float = 0.8,
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
