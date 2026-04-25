"""Tests for the live execution wiring."""

from __future__ import annotations

from typing import Any

import pytest

from tests.prompt_bench.live_runner import make_run_one
from tests.prompt_bench.scenarios import Scenario, ScenarioTurn
from tests.prompt_bench.variants import PromptOverlay


def _scenario(id_: str = "s") -> Scenario:
    return Scenario(
        id=id_,
        target="reference_deep_agent.core_identity",
        turns=[ScenarioTurn(user="hello")],
        samples=1,
    )


class TestMakeRunOne:
    async def test_smoke_invokes_build_and_drive(self) -> None:
        build_called: list[Any] = []
        drive_called: list[Any] = []

        def fake_build(overlay: PromptOverlay, deps: Any) -> str:
            build_called.append((overlay.name, deps))
            return "fake-graph"

        async def fake_drive(graph: Any, scenario: Scenario) -> dict[str, Any]:
            drive_called.append((graph, scenario.id))
            return {"final_output": "answer", "tool_calls": [{"name": "t"}]}

        run_one = make_run_one(
            profile_name="reference_deep_agent",
            executor_llm=object(),
            deps_factory=lambda: ("ckpt", "store"),
            build_graph=fake_build,
            drive_scenario=fake_drive,
        )

        sample = await run_one(_scenario(), PromptOverlay(name="v"), 0)
        assert build_called == [("v", ("ckpt", "store"))]
        assert drive_called == [("fake-graph", "s")]
        assert sample.final_output == "answer"
        assert sample.tool_calls == [{"name": "t"}]
        assert sample.error is None
        assert sample.duration_ms >= 0

    async def test_exception_in_drive_surfaces_as_error(self) -> None:
        def fake_build(_overlay: PromptOverlay, _deps: Any) -> str:
            return "fake-graph"

        async def fake_drive(_graph: Any, _scenario: Scenario) -> dict[str, Any]:
            msg = "boom"
            raise RuntimeError(msg)

        run_one = make_run_one(
            profile_name="reference_deep_agent",
            executor_llm=object(),
            deps_factory=lambda: ("ckpt", "store"),
            build_graph=fake_build,
            drive_scenario=fake_drive,
        )

        sample = await run_one(_scenario(), PromptOverlay(name="v"), 0)
        assert sample.error == "RuntimeError"
        assert sample.final_output == ""
        assert sample.duration_ms >= 0

    async def test_middleware_patch_active_during_drive(self) -> None:
        """Overlay's middleware patch must be active while the graph runs.

        The reference agent's middleware reads ``_EXTRACTION_PROMPT``
        each turn — patching only at build time would be too narrow.
        We verify by checking the patched value from inside ``drive``.
        """
        from langgraph_kit.core.memory import extraction

        seen: list[str] = []

        def fake_build(_overlay: PromptOverlay, _deps: Any) -> str:
            return "fake-graph"

        async def fake_drive(_graph: Any, _scenario: Scenario) -> dict[str, Any]:
            seen.append(extraction._EXTRACTION_PROMPT)
            return {"final_output": "ok", "tool_calls": []}

        overlay = PromptOverlay(
            name="patched",
            middleware_overrides={
                "langgraph_kit.core.memory.extraction:_EXTRACTION_PROMPT": "PATCHED_VALUE",
            },
        )
        run_one = make_run_one(
            profile_name="reference_deep_agent",
            executor_llm=object(),
            deps_factory=lambda: ("ckpt", "store"),
            build_graph=fake_build,
            drive_scenario=fake_drive,
        )

        await run_one(_scenario(), overlay, 0)
        assert seen == ["PATCHED_VALUE"]
        # Restored on exit
        assert extraction._EXTRACTION_PROMPT != "PATCHED_VALUE"

    async def test_returns_bench_sample_with_correct_metadata(self) -> None:
        def fake_build(_overlay: PromptOverlay, _deps: Any) -> str:
            return "fake-graph"

        async def fake_drive(_graph: Any, _scenario: Scenario) -> dict[str, Any]:
            return {"final_output": "x", "tool_calls": []}

        run_one = make_run_one(
            profile_name="reference_deep_agent",
            executor_llm=object(),
            deps_factory=lambda: ("ckpt", "store"),
            build_graph=fake_build,
            drive_scenario=fake_drive,
        )

        sample = await run_one(_scenario("scen-1"), PromptOverlay(name="vname"), 7)
        assert sample.scenario_id == "scen-1"
        assert sample.sample_index == 7
        assert sample.overlay_name == "vname"


class TestDefaultDriveScenario:
    async def test_walks_turns_and_collects_final_ai_message(self) -> None:
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
            HumanMessage,
        )

        from tests.prompt_bench.live_runner import default_drive_scenario

        # Fake graph that records ainvokes and returns a growing
        # message list with a final AIMessage and a tool call.
        ainvoke_calls: list[Any] = []

        class _FakeGraph:
            def __init__(self) -> None:
                self.history: list[Any] = []

            async def ainvoke(
                self, payload: dict[str, Any], config: Any
            ) -> dict[str, Any]:
                ainvoke_calls.append((payload, config))
                self.history = list(payload["messages"])
                self.history.append(
                    AIMessage(
                        content="answer",
                        tool_calls=[
                            {"id": "call_1", "name": "tool_x", "args": {"k": "v"}}
                        ],
                    )
                )
                return {"messages": self.history}

        scenario = Scenario(
            id="s",
            target="t",
            turns=[ScenarioTurn(user="q1"), ScenarioTurn(user="q2")],
            samples=1,
        )
        result = await default_drive_scenario(_FakeGraph(), scenario)

        assert result["final_output"] == "answer"
        # Each turn invokes the graph once
        assert len(ainvoke_calls) == 2
        # First payload has 1 human message, second has 2 + AI from turn 1
        assert isinstance(ainvoke_calls[0][0]["messages"][0], HumanMessage)
        # Tool calls from BOTH turns are captured (default_drive_scenario
        # walks the full message list each time and re-records calls);
        # we accept that for Phase 0.5 since the agent under test is the
        # source of truth for tool-call dedup.
        assert len(result["tool_calls"]) >= 1
        assert result["tool_calls"][0]["name"] == "tool_x"


class TestUnknownProfile:
    async def test_unknown_profile_raises_at_build_time(self) -> None:
        # The default build_graph routes through profiles.build_profile_graph,
        # which raises KeyError for unknown profile names.
        run_one = make_run_one(
            profile_name="totally_unknown_profile",
            executor_llm=object(),
            deps_factory=lambda: ("ckpt", "store"),
            drive_scenario=_unused_drive_scenario,
        )
        sample = await run_one(_scenario(), PromptOverlay(name="v"), 0)
        # Failure is captured as error= on the sample
        assert sample.error == "KeyError"


async def _unused_drive_scenario(_graph: Any, _scenario: Scenario) -> dict[str, Any]:
    pytest.fail("drive_scenario should not be reached if build_graph fails")
    return {}
