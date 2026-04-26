"""Tests for ``langgraph_kit.contrib.watchers`` (issue #82)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgraph_kit.contrib.watchers import (
    StoreWatcherRegistry,
    StoreWatcherRunner,
    StoreWatcherSpec,
)
from langgraph_kit.testing import FakeStore

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


def _spec(
    spec_id: str = "alert-batcher",
    namespace: tuple[str, ...] = ("alerts", "unhandled"),
    threshold: int = 3,
    payload: str = "Process the queue.",
) -> StoreWatcherSpec:
    """Default-shaped spec for tests; predicate fires when len >= threshold."""
    return StoreWatcherSpec(
        id=spec_id,
        agent_id="batcher",
        namespace=namespace,
        predicate=lambda items: len(items) >= threshold,
        poll_interval_seconds=0.05,
        payload_template=payload,
    )


# ---------------------------------------------------------------------------
# StoreWatcherRegistry
# ---------------------------------------------------------------------------


class TestStoreWatcherRegistry:
    def test_register_and_get_round_trip(self) -> None:
        registry = StoreWatcherRegistry()
        spec = _spec()
        registry.register(spec)
        assert registry.get(spec.id) is spec

    def test_register_replaces_existing_id(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(spec_id="x"))
        spec_b = _spec(spec_id="x", payload="updated")
        registry.register(spec_b)
        assert registry.get("x") is spec_b

    def test_remove_drops_spec(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(spec_id="x"))
        registry.remove("x")
        assert registry.get("x") is None

    def test_register_validates_poll_interval(self) -> None:
        registry = StoreWatcherRegistry()
        with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
            registry.register(
                StoreWatcherSpec(
                    id="bad",
                    agent_id="a",
                    namespace=("ns",),
                    predicate=lambda _: True,
                    poll_interval_seconds=0,
                )
            )

    def test_list_ids_and_all_specs_match(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(spec_id="a"))
        registry.register(_spec(spec_id="b"))
        assert sorted(registry.list_ids()) == ["a", "b"]
        assert {s.id for s in registry.all_specs()} == {"a", "b"}


# ---------------------------------------------------------------------------
# StoreWatcherRunner — edge-trigger semantics via poll_now
# ---------------------------------------------------------------------------


class TestStoreWatcherRunner:
    @pytest.mark.asyncio
    async def test_predicate_unmet_does_not_fire(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(threshold=10))
        store = FakeStore()
        await store.aput(("alerts", "unhandled"), "k1", {"x": 1})
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            fired = await runner.poll_now("alert-batcher")

        assert fired is False
        assert graph.calls == []

    @pytest.mark.asyncio
    async def test_predicate_met_fires_once(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(threshold=2))
        store = FakeStore()
        for i in range(3):
            await store.aput(("alerts", "unhandled"), f"k{i}", {"i": i})
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            fired = await runner.poll_now("alert-batcher")

        assert fired is True
        assert len(graph.calls) == 1
        msg = graph.calls[0][0]["messages"][0]
        assert msg.content == "Process the queue."

    @pytest.mark.asyncio
    async def test_stable_true_state_does_not_re_fire(self) -> None:
        """Once armed, repeated ``poll_now`` calls don't re-fire until predicate flips false."""
        registry = StoreWatcherRegistry()
        registry.register(_spec(threshold=2))
        store = FakeStore()
        for i in range(3):
            await store.aput(("alerts", "unhandled"), f"k{i}", {"i": i})
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            f1 = await runner.poll_now("alert-batcher")
            f2 = await runner.poll_now("alert-batcher")
            f3 = await runner.poll_now("alert-batcher")

        assert (f1, f2, f3) == (True, False, False)
        assert len(graph.calls) == 1

    @pytest.mark.asyncio
    async def test_predicate_flip_re_arms_for_next_rising_edge(self) -> None:
        """false → true → false → true should fire twice."""
        registry = StoreWatcherRegistry()
        registry.register(_spec(threshold=2))
        store = FakeStore()
        # Start: 0 items → predicate false.
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            # Poll 1: false (no items).
            assert await runner.poll_now("alert-batcher") is False
            # Add items → poll 2: true (rising edge).
            await store.aput(("alerts", "unhandled"), "k1", {"i": 1})
            await store.aput(("alerts", "unhandled"), "k2", {"i": 2})
            assert await runner.poll_now("alert-batcher") is True
            # Stable true → poll 3: still armed, no fire.
            assert await runner.poll_now("alert-batcher") is False
            # Drop below threshold → poll 4: predicate false, re-arms.
            await store.adelete(("alerts", "unhandled"), "k1")
            await store.adelete(("alerts", "unhandled"), "k2")
            assert await runner.poll_now("alert-batcher") is False
            # Back above → poll 5: rising edge, fires again.
            for i in range(3):
                await store.aput(("alerts", "unhandled"), f"new{i}", {"i": i})
            assert await runner.poll_now("alert-batcher") is True

        assert len(graph.calls) == 2

    @pytest.mark.asyncio
    async def test_poll_now_unknown_spec_raises(self) -> None:
        registry = StoreWatcherRegistry()
        store = FakeStore()
        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: _FakeGraph()
        ) as runner:
            with pytest.raises(KeyError, match="Unknown watcher spec"):
                await runner.poll_now("missing")

    @pytest.mark.asyncio
    async def test_predicate_exception_does_not_kill_watcher(self) -> None:
        """Bad predicate logs + treats as False so the loop survives."""

        def bad_predicate(_items: list[Any]) -> bool:
            msg = "boom"
            raise RuntimeError(msg)

        registry = StoreWatcherRegistry()
        registry.register(
            StoreWatcherSpec(
                id="buggy",
                agent_id="a",
                namespace=("alerts", "unhandled"),
                predicate=bad_predicate,
                poll_interval_seconds=0.05,
            )
        )
        store = FakeStore()
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            fired = await runner.poll_now("buggy")

        assert fired is False
        assert graph.calls == []  # Predicate raised → treated as False → no fire.


# ---------------------------------------------------------------------------
# Lifecycle (start / stop spawn + cancel polling tasks)
# ---------------------------------------------------------------------------


class TestRunnerLifecycle:
    @pytest.mark.asyncio
    async def test_start_spawns_one_task_per_spec(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec(spec_id="a"))
        registry.register(_spec(spec_id="b"))
        runner = StoreWatcherRunner(
            registry,
            store=FakeStore(),
            graph_resolver=lambda _: _FakeGraph(),
        )
        await runner.start()
        try:
            assert len(runner._tasks) == 2
        finally:
            await runner.stop()
        assert runner._tasks == {}

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec())
        runner = StoreWatcherRunner(
            registry,
            store=FakeStore(),
            graph_resolver=lambda _: _FakeGraph(),
        )
        await runner.start()
        await runner.start()  # Must not raise.
        await runner.stop()

    @pytest.mark.asyncio
    async def test_aexit_stops_tasks(self) -> None:
        registry = StoreWatcherRegistry()
        registry.register(_spec())
        runner = StoreWatcherRunner(
            registry,
            store=FakeStore(),
            graph_resolver=lambda _: _FakeGraph(),
        )
        async with runner:
            await runner.start()
            assert len(runner._tasks) == 1
        # After exit, tasks are cancelled and dict cleared.
        assert runner._tasks == {}

    @pytest.mark.asyncio
    async def test_loop_actually_polls_and_fires(self) -> None:
        """End-to-end: start the loop, populate the store, observe the fire."""
        registry = StoreWatcherRegistry()
        # Short poll interval so the test doesn't hang.
        registry.register(_spec(threshold=2))
        store = FakeStore()
        # Pre-populate so the first poll fires.
        for i in range(3):
            await store.aput(("alerts", "unhandled"), f"k{i}", {"i": i})
        graph = _FakeGraph()

        async with StoreWatcherRunner(
            registry, store=store, graph_resolver=lambda _: graph
        ) as runner:
            await runner.start()
            # Wait long enough for at least one poll cycle (interval=0.05s).
            await asyncio.sleep(0.2)

        assert len(graph.calls) >= 1
