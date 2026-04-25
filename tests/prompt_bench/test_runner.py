"""Tests for the bench runner — iteration logic only.

Profile-specific ``run_one`` callables live in :mod:`tests.prompt_bench.profiles`
and ``conftest.py``; those are tested separately. This file only covers
the trivial scenarios x samples loop that ``BenchRunner`` itself owns.
"""

from __future__ import annotations

from tests.prompt_bench.runner import BenchRunner, BenchSample
from tests.prompt_bench.scenarios import Scenario, ScenarioTurn
from tests.prompt_bench.variants import PromptOverlay


def _scenario(id_: str, samples: int = 3) -> Scenario:
    return Scenario(
        id=id_,
        target="t",
        turns=[ScenarioTurn(user="hi")],
        samples=samples,
    )


class TestBenchRunner:
    async def test_iterates_scenarios_x_samples(self) -> None:
        seen: list[tuple[str, int]] = []

        async def run_one(
            scenario: Scenario, overlay: PromptOverlay, idx: int
        ) -> BenchSample:
            seen.append((scenario.id, idx))
            return BenchSample(
                scenario_id=scenario.id,
                sample_index=idx,
                overlay_name=overlay.name,
                duration_ms=1.0,
                final_output="ok",
            )

        runner = BenchRunner(run_one)
        report = await runner.run(
            scenarios=[_scenario("a", samples=2), _scenario("b", samples=3)],
            overlay=PromptOverlay(name="v"),
        )

        assert report.overlay_name == "v"
        assert len(report.samples) == 5
        assert seen == [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)]

    async def test_propagates_run_one_errors_into_sample(self) -> None:
        async def run_one(
            scenario: Scenario, overlay: PromptOverlay, idx: int
        ) -> BenchSample:
            return BenchSample(
                scenario_id=scenario.id,
                sample_index=idx,
                overlay_name=overlay.name,
                duration_ms=0.0,
                final_output="",
                error="kaboom",
            )

        runner = BenchRunner(run_one)
        report = await runner.run(
            scenarios=[_scenario("a", samples=1)],
            overlay=PromptOverlay(name="v"),
        )
        assert len(report.samples) == 1
        assert report.samples[0].error == "kaboom"

    async def test_traces_method_returns_trace_data(self) -> None:
        async def run_one(
            scenario: Scenario, overlay: PromptOverlay, idx: int
        ) -> BenchSample:
            return BenchSample(
                scenario_id=scenario.id,
                sample_index=idx,
                overlay_name=overlay.name,
                duration_ms=10.0,
                final_output="answer",
                tool_calls=[{"name": "memory_save", "args": {}}],
            )

        runner = BenchRunner(run_one)
        report = await runner.run(
            scenarios=[_scenario("a", samples=1)],
            overlay=PromptOverlay(name="v"),
        )
        traces = report.traces()
        assert len(traces) == 1
        assert traces[0].output == "answer"
        assert traces[0].metadata["tool_calls"] == 1
        assert traces[0].metadata["tools_used"] == ["memory_save"]
        assert traces[0].metadata["status"] == "ok"
