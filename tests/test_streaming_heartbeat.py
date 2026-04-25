"""Coverage — SSE heartbeats + ``id:`` lines added by issue #11.

`stream_agent_events` now:

- prefixes every emitted chunk with an SSE ``id: <n>`` line so a
  reconnecting client can hand back the last id via ``Last-Event-ID``
- emits a ``{"heartbeat": {"ts": ..., "last_event_id": ...}}`` chunk
  every ``heartbeat_interval`` seconds during quiet periods so
  proxies / load balancers don't drop the idle connection
- accepts ``heartbeat_interval=None`` to disable heartbeats
- preserves the sequence across heartbeats (heartbeat last_event_id
  reflects the most recently emitted real event id)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

import pytest


def _data_lines(chunk: str) -> list[str]:
    """Extract every ``data:`` line from a chunk."""
    return [
        line.removeprefix("data: ").strip()
        for line in chunk.split("\n")
        if line.startswith("data: ")
    ]


def _id_lines(chunk: str) -> list[str]:
    return [
        line.removeprefix("id: ").strip()
        for line in chunk.split("\n")
        if line.startswith("id: ")
    ]


class _StubGraph:
    """Minimal graph implementing the subset of astream_events the streamer uses."""

    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._events = events or []

    async def astream_events(
        self, _input: Any, config: Any = None, version: str = "v2"
    ) -> Any:  # pragma: no cover — version arg not used in tests
        _ = config, version
        for ev in self._events:
            yield ev

    async def aget_state(self, _config: Any) -> Any:
        class _S:
            values: ClassVar[dict[str, Any]] = {}
            tasks: ClassVar[list[Any]] = []

        return _S()


class _SlowStubGraph:
    """astream_events that hangs forever — useful to test heartbeats."""

    async def astream_events(
        self, _input: Any, config: Any = None, version: str = "v2"
    ) -> Any:
        _ = config, version
        # Wait long enough for the test to capture multiple heartbeats.
        await asyncio.sleep(60)
        if False:
            yield None  # pragma: no cover — make this an async generator

    async def aget_state(self, _config: Any) -> Any:
        class _S:
            values: ClassVar[dict[str, Any]] = {}
            tasks: ClassVar[list[Any]] = []

        return _S()


# ---------------------------------------------------------------------------
# Event-id semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_chunk_carries_an_id_line() -> None:
    """Every emitted SSE chunk must include an ``id: N`` line."""
    from langgraph_kit.streaming import stream_agent_events

    chunks: list[str] = []
    graph = _StubGraph()  # No events -> just the closing [DONE]
    async for chunk in stream_agent_events(graph, {}, {}, heartbeat_interval=None):
        chunks.append(chunk)

    assert len(chunks) >= 1
    for chunk in chunks:
        ids = _id_lines(chunk)
        assert len(ids) == 1, f"chunk missing id line: {chunk!r}"
        # All ids must be integers parseable as a sequence number.
        int(ids[0])


@pytest.mark.asyncio
async def test_event_ids_are_monotonic_starting_from_zero() -> None:
    from langgraph_kit.streaming import stream_agent_events

    chunks: list[str] = []
    graph = _StubGraph()
    async for chunk in stream_agent_events(graph, {}, {}, heartbeat_interval=None):
        chunks.append(chunk)

    seen_ids = [int(_id_lines(c)[0]) for c in chunks]
    assert seen_ids == sorted(seen_ids)
    assert seen_ids[0] == 0


@pytest.mark.asyncio
async def test_done_sentinel_carries_an_id() -> None:
    """``[DONE]`` is just another chunk — it gets an id like everything else."""
    from langgraph_kit.streaming import stream_agent_events

    graph = _StubGraph()
    chunks: list[str] = []
    async for chunk in stream_agent_events(graph, {}, {}, heartbeat_interval=None):
        chunks.append(chunk)

    done_chunk = next(c for c in chunks if "[DONE]" in c)
    assert _id_lines(done_chunk), "DONE chunk missing id line"
    assert _data_lines(done_chunk) == ["[DONE]"]


# ---------------------------------------------------------------------------
# Heartbeat behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_emitted_during_quiet_stream() -> None:
    """A stream that produces no events for longer than the heartbeat
    interval must emit at least one heartbeat chunk."""
    from langgraph_kit.streaming import stream_agent_events

    graph = _SlowStubGraph()
    chunks: list[str] = []

    async def _drive() -> None:
        async for chunk in stream_agent_events(graph, {}, {}, heartbeat_interval=0.05):
            chunks.append(chunk)
            heartbeat_seen = any("heartbeat" in c for c in chunks)
            if heartbeat_seen and len(chunks) >= 2:
                break

    # Cap the test so a regression doesn't hang the suite.
    await asyncio.wait_for(_drive(), timeout=2.0)

    heartbeats = [c for c in chunks if "heartbeat" in c]
    assert heartbeats, f"expected at least one heartbeat in {chunks!r}"
    payload = json.loads(_data_lines(heartbeats[0])[0])
    assert "heartbeat" in payload
    assert "ts" in payload["heartbeat"]
    assert "last_event_id" in payload["heartbeat"]


@pytest.mark.asyncio
async def test_heartbeat_disabled_when_interval_is_none() -> None:
    """Setting ``heartbeat_interval=None`` produces no heartbeat chunks."""
    from langgraph_kit.streaming import stream_agent_events

    graph = _StubGraph()  # finishes quickly
    chunks: list[str] = []
    async for chunk in stream_agent_events(graph, {}, {}, heartbeat_interval=None):
        chunks.append(chunk)
    assert not any("heartbeat" in c for c in chunks)


@pytest.mark.asyncio
async def test_heartbeat_carries_correct_last_event_id() -> None:
    """A heartbeat after several real events should report the id of the
    last real chunk emitted (not its own id)."""
    from langgraph_kit.streaming import stream_agent_events

    # First emit one short event, then hang. The heartbeat that arrives
    # should have last_event_id = id of the [DONE] preceding it... but
    # that means we never reach quiescence. Use a graph that yields one
    # tool_start then hangs.

    class _OneEventThenHang:
        async def astream_events(
            self, _input: Any, config: Any = None, version: str = "v2"
        ) -> Any:
            _ = config, version
            yield {
                "event": "on_tool_start",
                "data": {"input": {"x": 1}},
                "name": "do_thing",
                "run_id": "rid-1",
            }
            await asyncio.sleep(60)
            if False:
                yield None  # pragma: no cover

        async def aget_state(self, _config: Any) -> Any:
            class _S:
                values: ClassVar[dict[str, Any]] = {}
                tasks: ClassVar[list[Any]] = []

            return _S()

    chunks: list[str] = []

    async def _drive() -> None:
        async for chunk in stream_agent_events(
            _OneEventThenHang(), {}, {}, heartbeat_interval=0.05
        ):
            chunks.append(chunk)
            if any("heartbeat" in c for c in chunks):
                break

    await asyncio.wait_for(_drive(), timeout=2.0)

    real_event_chunks = [c for c in chunks if "tool_call_start" in c]
    heartbeat_chunks = [c for c in chunks if "heartbeat" in c]
    assert real_event_chunks
    assert heartbeat_chunks

    real_id = int(_id_lines(real_event_chunks[0])[0])
    heartbeat_payload = json.loads(_data_lines(heartbeat_chunks[0])[0])
    # The heartbeat's reported last_event_id should reflect the id of
    # the most recently emitted real chunk, not the heartbeat itself.
    assert heartbeat_payload["heartbeat"]["last_event_id"] == real_id


# ---------------------------------------------------------------------------
# Default interval is the public constant
# ---------------------------------------------------------------------------


def test_default_heartbeat_interval_is_15_seconds() -> None:
    from langgraph_kit.streaming import DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    assert DEFAULT_HEARTBEAT_INTERVAL_SECONDS == 15.0
