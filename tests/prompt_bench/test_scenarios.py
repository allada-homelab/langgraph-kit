"""Tests for the scenario YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.prompt_bench.scenarios import (
    Scenario,
    discover_scenarios,
    load_scenario,
)


class TestLoadScenario:
    def test_loads_minimal_scenario(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "id": "minimal",
                    "target": "reference_deep_agent.core_identity",
                    "turns": [{"user": "hello"}],
                }
            )
        )
        scenario = load_scenario(path)
        assert isinstance(scenario, Scenario)
        assert scenario.id == "minimal"
        assert scenario.samples == 5
        assert scenario.expected_behaviors.must_call_tool == []

    def test_rejects_empty_turns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "id": "bad",
                    "target": "x",
                    "turns": [],
                }
            )
        )
        with pytest.raises(ValueError, match="at least one user turn"):
            load_scenario(path)

    def test_rejects_zero_samples(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "id": "bad",
                    "target": "x",
                    "turns": [{"user": "hi"}],
                    "samples": 0,
                }
            )
        )
        with pytest.raises(ValueError, match="samples must be"):
            load_scenario(path)

    def test_rejects_extra_fields(self, tmp_path: Path) -> None:
        """Strict schema — extra keys cause a clear error rather than silently dropping."""
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "id": "bad",
                    "target": "x",
                    "turns": [{"user": "hi"}],
                    "extra_typo_field": "oops",
                }
            )
        )
        with pytest.raises(ValueError, match="extra_typo_field"):
            load_scenario(path)


class TestDiscoverScenarios:
    def test_discovers_all_under_root(self) -> None:
        # The shipped seed scenarios — a stable invariant
        root = Path(__file__).parent
        scenarios = discover_scenarios(root)
        # 12 seed scenarios across 4 targets
        assert len(scenarios) >= 12
        # All have non-empty IDs
        assert all(s.id for s in scenarios)

    def test_filters_by_target(self) -> None:
        root = Path(__file__).parent
        ref_only = discover_scenarios(root, target="reference_deep_agent")
        for s in ref_only:
            assert s.agent_profile == "reference_deep_agent" or s.target.startswith(
                "reference_deep_agent"
            )

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        # Empty root has no scenarios/ subdir
        assert discover_scenarios(tmp_path) == []
