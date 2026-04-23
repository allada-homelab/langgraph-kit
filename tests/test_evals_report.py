"""Coverage fill — eval report serialization + console printing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langgraph_kit.evals.models import EvalReport, MetricSummary
from langgraph_kit.evals.report import print_console_report, write_json_report

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _report() -> EvalReport:
    return EvalReport(
        timestamp="2026-04-24T00:00:00Z",
        model="test-model",
        duration_seconds=1.5,
        total_traces=2,
        metrics={
            "pass_metric": MetricSummary(
                name="pass_metric",
                data_type="BOOLEAN",
                count=2,
                pass_rate=1.0,
                values=[True, True],
            ),
            "warn_metric": MetricSummary(
                name="warn_metric",
                data_type="NUMERIC",
                count=2,
                mean=0.6,
                pass_rate=0.6,
                values=[0.5, 0.7],
            ),
            "fail_metric": MetricSummary(
                name="fail_metric",
                data_type="BOOLEAN",
                count=2,
                pass_rate=0.0,
                values=[False, False],
            ),
        },
        trace_results=[{"trace_id": "t1"}, {"trace_id": "t2"}],
    )


def test_write_json_report_creates_parent_dirs_and_writes(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "report.json"
    write_json_report(_report(), path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["total_traces"] == 2
    assert "pass_metric" in loaded["metrics"]
    # Parent dir was auto-created.
    assert path.parent.is_dir()


def test_print_console_report_tags_pass_warn_fail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_console_report(_report(), pass_threshold=0.8, warn_threshold=0.5)
    output = capsys.readouterr().out
    assert "Evaluation Report" in output
    assert "Traces evaluated: 2" in output
    # One metric per status tier — output labels should reflect that.
    assert "PASS" in output
    assert "WARN" in output
    assert "FAIL" in output
    # pass_rate formatting is percent with one decimal.
    assert "100.0%" in output


def test_print_console_report_omits_mean_when_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalReport(
        timestamp="now",
        total_traces=1,
        metrics={
            "bare": MetricSummary(
                name="bare",
                data_type="BOOLEAN",
                count=1,
            )
        },
    )
    print_console_report(report)
    output = capsys.readouterr().out
    assert "bare" in output
    # When mean is None no "mean=" clause appears.
    assert "mean=" not in output
