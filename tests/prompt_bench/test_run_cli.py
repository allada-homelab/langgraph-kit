"""Smoke tests for the bench CLI.

These tests exercise the CLI subcommands (``list-targets``,
``list-scenarios``, ``run``, ``diff``) against the deterministic stub
LLM. The ``signal-check`` subcommand is exercised end-to-end via the
nightly workflow against a real LLM — too expensive for unit tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tests.prompt_bench.run import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestListCommands:
    def test_list_targets(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-targets"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "reference_deep_agent.core_identity" in out
        assert "memory_extraction.prompt" in out

    def test_list_scenarios_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-scenarios"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "12 scenarios" in out

    def test_list_scenarios_filtered(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-scenarios", "--target", "reference_deep_agent"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "scenarios:" in out


class TestDiffCommand:
    """`diff` is independent of the agent build path and is fully
    exercisable with hand-crafted JSON reports."""

    def test_diff_unanimous_tie_passes_zero_signal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Two identical reports → 0 wins, all ties under the stub judge
        report_a = {
            "overlay_name": "a",
            "samples": [
                {
                    "scenario_id": "s1",
                    "sample_index": i,
                    "overlay_name": "a",
                    "duration_ms": 100.0,
                    "final_output": "answer",
                    "tool_calls": [],
                    "error": None,
                }
                for i in range(2)
            ],
        }
        report_b = {**report_a, "overlay_name": "b"}
        for sample in report_b["samples"]:
            sample["overlay_name"] = "b"

        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(report_a))
        b_path.write_text(json.dumps(report_b))

        out_path = tmp_path / "diff.md"
        rc = main(
            [
                "diff",
                "--base",
                str(a_path),
                "--variant",
                str(b_path),
                "--out",
                str(out_path),
            ]
        )
        out = capsys.readouterr().out
        # Stub judges return "tie" → no variant wins → win_rate = 0% → fails 60% bar
        assert "**0.0%**" in out
        assert "BELOW 60% bar" in out
        assert rc != 0
        assert out_path.is_file()
        assert (out_path.with_suffix(".json")).is_file()


class TestUsageErrors:
    def test_run_unknown_target(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["run", "--target", "nonexistent_target", "--variant", "baseline"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "nonexistent_target" in err

    def test_run_unknown_variant(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(
            [
                "run",
                "--target",
                "reference_deep_agent.core_identity",
                "--variant",
                "nonexistent_variant",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "nonexistent_variant" in err

    def test_signal_check_unknown_target(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["signal-check", "--target", "nope"])
        assert rc == 2
