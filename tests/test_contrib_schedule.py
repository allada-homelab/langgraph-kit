"""Tests for ``langgraph_kit.contrib.schedule`` (issue #81)."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.contrib.schedule import (
    ScheduledRegistry,
    ScheduledSpec,
    ScheduledTriggerRunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Records each ``ainvoke`` call so tests can assert on what was sent."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((input_data, config))
        return {"messages": []}


# ---------------------------------------------------------------------------
# ScheduledSpec
# ---------------------------------------------------------------------------


class TestScheduledSpec:
    def test_default_payload_is_empty(self) -> None:
        spec = ScheduledSpec(id="weekly", agent_id="agent", cron="0 9 * * MON")
        assert spec.payload_template == ""
        assert spec.payload_data == {}

    def test_spec_is_frozen(self) -> None:
        spec = ScheduledSpec(id="weekly", agent_id="agent", cron="0 9 * * MON")
        with pytest.raises(Exception):  # noqa: B017,PT011
            spec.cron = "* * * * *"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScheduledRegistry
# ---------------------------------------------------------------------------


class TestScheduledRegistry:
    def test_register_and_get_round_trip(self) -> None:
        registry = ScheduledRegistry()
        spec = ScheduledSpec(id="weekly", agent_id="agent", cron="0 9 * * MON")
        registry.register(spec)
        assert registry.get("weekly") is spec

    def test_get_unknown_id_returns_none(self) -> None:
        assert ScheduledRegistry().get("missing") is None

    def test_register_replaces_existing_id(self) -> None:
        registry = ScheduledRegistry()
        registry.register(ScheduledSpec(id="x", agent_id="a", cron="0 9 * * *"))
        registry.register(ScheduledSpec(id="x", agent_id="a", cron="0 10 * * *"))
        spec = registry.get("x")
        assert spec is not None
        assert spec.cron == "0 10 * * *"

    def test_remove_drops_spec(self) -> None:
        registry = ScheduledRegistry()
        registry.register(ScheduledSpec(id="x", agent_id="a", cron="0 9 * * *"))
        registry.remove("x")
        assert registry.get("x") is None

    def test_list_ids_and_all_specs_match(self) -> None:
        registry = ScheduledRegistry()
        registry.register(ScheduledSpec(id="a", agent_id="x", cron="0 * * * *"))
        registry.register(ScheduledSpec(id="b", agent_id="x", cron="*/5 * * * *"))
        assert sorted(registry.list_ids()) == ["a", "b"]
        assert {s.id for s in registry.all_specs()} == {"a", "b"}

    def test_invalid_cron_raises_at_registration(self) -> None:
        """Catches typos at deploy time, not at first fire."""
        registry = ScheduledRegistry()
        with pytest.raises(ValueError, match=r"Wrong number of fields|Invalid"):
            registry.register(ScheduledSpec(id="bad", agent_id="a", cron="not a cron"))


# ---------------------------------------------------------------------------
# ScheduledTriggerRunner
# ---------------------------------------------------------------------------


class TestScheduledTriggerRunner:
    @pytest.mark.asyncio
    async def test_fire_now_invokes_agent_with_payload(self) -> None:
        spec = ScheduledSpec(
            id="weekly",
            agent_id="reports",
            cron="0 9 * * MON",
            payload_template="Generate this week's summary report.",
        )
        registry = ScheduledRegistry()
        registry.register(spec)
        graph = _FakeGraph()

        async with ScheduledTriggerRunner(
            registry, graph_resolver=lambda _: graph
        ) as runner:
            thread_id = await runner.fire_now("weekly")

        assert thread_id.startswith("scheduled-weekly-")
        assert len(graph.calls) == 1
        input_data, config = graph.calls[0]
        msg = input_data["messages"][0]
        assert msg.content == "Generate this week's summary report."
        assert config is not None
        assert config["configurable"]["thread_id"] == thread_id

    @pytest.mark.asyncio
    async def test_fire_now_unknown_spec_raises_keyerror(self) -> None:
        registry = ScheduledRegistry()
        async with ScheduledTriggerRunner(
            registry, graph_resolver=lambda _: _FakeGraph()
        ) as runner:
            with pytest.raises(KeyError, match="Unknown scheduled spec"):
                await runner.fire_now("missing")

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        """Idempotent ``start()`` + ``stop()`` matches the lifespan pattern."""
        spec = ScheduledSpec(id="job", agent_id="agent", cron="* * * * *")
        registry = ScheduledRegistry()
        registry.register(spec)

        runner = ScheduledTriggerRunner(registry, graph_resolver=lambda _: _FakeGraph())
        await runner.start()
        # Second start is a no-op (idempotent).
        await runner.start()
        await runner.stop()
        # Stop is also idempotent.
        await runner.stop()

    @pytest.mark.asyncio
    async def test_two_fires_get_distinct_thread_ids(self) -> None:
        spec = ScheduledSpec(id="job", agent_id="agent", cron="* * * * *")
        registry = ScheduledRegistry()
        registry.register(spec)
        graph = _FakeGraph()

        async with ScheduledTriggerRunner(
            registry, graph_resolver=lambda _: graph
        ) as runner:
            t1 = await runner.fire_now("job")
            t2 = await runner.fire_now("job")

        assert t1 != t2
        assert len(graph.calls) == 2

    @pytest.mark.asyncio
    async def test_aenter_aexit_propagates_stop(self) -> None:
        """Context manager exit calls stop() even when no exception was raised."""
        registry = ScheduledRegistry()
        registry.register(ScheduledSpec(id="job", agent_id="a", cron="* * * * *"))
        runner = ScheduledTriggerRunner(registry, graph_resolver=lambda _: _FakeGraph())
        async with runner:
            await runner.start()
            assert runner._scheduler is not None
            assert runner._scheduler.running
        # After exit, scheduler is stopped + cleared.
        assert runner._scheduler is None

    @pytest.mark.asyncio
    async def test_fire_with_unregistered_spec_logs_and_returns_empty(self) -> None:
        """Race: spec removed after the scheduler queued it. Don't crash."""
        spec = ScheduledSpec(id="job", agent_id="agent", cron="* * * * *")
        registry = ScheduledRegistry()
        registry.register(spec)

        async with ScheduledTriggerRunner(
            registry, graph_resolver=lambda _: _FakeGraph()
        ) as runner:
            # Simulate the race by calling the internal _fire after removal.
            registry.remove("job")
            result = await runner._fire("job")
            assert result == ""

    @pytest.mark.asyncio
    async def test_graph_resolver_receives_agent_id(self) -> None:
        """Resolver is called with spec.agent_id (not spec.id)."""
        seen: list[str] = []

        def resolver(agent_id: str) -> _FakeGraph:
            seen.append(agent_id)
            return _FakeGraph()

        registry = ScheduledRegistry()
        registry.register(
            ScheduledSpec(
                id="schedule-x",
                agent_id="agent-y",
                cron="* * * * *",
            )
        )
        async with ScheduledTriggerRunner(registry, graph_resolver=resolver) as runner:
            await runner.fire_now("schedule-x")

        assert seen == ["agent-y"]
