"""Bench runner — execute scenarios under a given prompt overlay.

The runner is intentionally trivial: it iterates scenarios x samples and
delegates the messy work (apply overlay, build graph, invoke
conversation, capture trace) to a profile-specific ``run_one``
callable.

This keeps profile-specific knowledge — how to translate a
``PromptOverlay`` into a built ``reference_deep_agent`` graph, how to
keep middleware patches active across both compile and invoke — in
one place (``profiles.py`` / ``conftest.py``) and out of the harness
core.

See ``profiles.build_profile_graph`` for the canonical wiring.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph_kit.evals.models import TraceData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.prompt_bench.scenarios import Scenario
    from tests.prompt_bench.variants import PromptOverlay


@dataclass
class BenchSample:
    """One execution of a scenario — captured turn-by-turn."""

    scenario_id: str
    sample_index: int
    overlay_name: str
    duration_ms: float
    final_output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_trace(self) -> TraceData:
        """Convert to the canonical :class:`TraceData` shape used by metrics."""
        return TraceData(
            id=f"{self.scenario_id}#{self.overlay_name}#{self.sample_index}",
            name=self.scenario_id,
            input={"scenario": self.scenario_id, "overlay": self.overlay_name},
            output=self.final_output,
            tags=[f"overlay:{self.overlay_name}", f"scenario:{self.scenario_id}"],
            duration_ms=self.duration_ms,
            metadata={
                "tool_calls": len(self.tool_calls),
                "tools_used": [c.get("name") for c in self.tool_calls],
                "tool_errors": sum(1 for c in self.tool_calls if c.get("error")),
                "error": self.error,
                "status": "error" if self.error else "ok",
            },
        )


@dataclass
class BenchReport:
    """Result of running a set of scenarios under a single overlay."""

    overlay_name: str
    samples: list[BenchSample] = field(default_factory=list)

    def traces(self) -> list[TraceData]:
        return [s.to_trace() for s in self.samples]


RunOne = Callable[
    ["Scenario", "PromptOverlay", int],
    Awaitable[BenchSample],
]
"""``async (scenario, overlay, sample_index) -> BenchSample``.

Profile-specific. Should:

1. Apply *overlay* (sections + middleware) for the duration of the call.
2. Build / reuse the agent graph for the scenario's profile.
3. Drive *scenario*'s turns through the graph.
4. Return a populated :class:`BenchSample`.

Errors should be caught and surfaced as ``error`` on the sample, not
re-raised — one bad scenario should not abort the whole bench run.
"""


class BenchRunner:
    """Iterates scenarios x samples, delegates to a profile-specific ``run_one``."""

    def __init__(self, run_one: RunOne) -> None:
        super().__init__()
        self._run_one = run_one

    async def run(
        self,
        scenarios: Sequence[Scenario],
        overlay: PromptOverlay,
    ) -> BenchReport:
        report = BenchReport(overlay_name=overlay.name)
        for scenario in scenarios:
            for i in range(scenario.samples):
                sample = await self._run_one(scenario, overlay, i)
                report.samples.append(sample)
        return report
