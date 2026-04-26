# pyright: reportPrivateUsage=false
# Strict-mode prep: this test probes ``_parse_sentinel`` by design.
"""Tests for streaming module — sentinel parsing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_kit.streaming import _parse_sentinel, stream_agent_events


class TestParseSentinel:
    def test_artifact_sentinel(self) -> None:
        from langgraph_kit.core.artifacts import ARTIFACT_SENTINEL

        result = _parse_sentinel(
            f'{ARTIFACT_SENTINEL}{{"type": "code", "title": "test"}}'
        )
        assert result is not None
        assert "artifact" in result
        assert result["artifact"]["type"] == "code"

    def test_progress_sentinel(self) -> None:
        from langgraph_kit.core.ui_events import PROGRESS_SENTINEL

        result = _parse_sentinel(f'{PROGRESS_SENTINEL}{{"step": 1, "total": 3}}')
        assert result is not None
        assert "progress" in result
        assert result["progress"]["step"] == 1

    def test_unknown_prefix_returns_none(self) -> None:
        result = _parse_sentinel("Just normal tool output")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _parse_sentinel("")
        assert result is None

    def test_malformed_json_returns_none(self) -> None:
        from langgraph_kit.core.artifacts import ARTIFACT_SENTINEL

        result = _parse_sentinel(f"{ARTIFACT_SENTINEL}{{not valid json}}")
        assert result is None


class _FakeGraph:
    """Minimal graph stub for stream_agent_events — emits a fixed event list."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(self, *_args: Any, **_kwargs: Any) -> Any:
        for ev in self._events:
            yield ev

    async def aget_state(self, *_args: Any, **_kwargs: Any) -> Any:
        class _S:
            def __init__(self) -> None:
                self.values: dict[str, Any] = {}
                self.tasks: list[Any] = []

        return _S()


def _token_event(text: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    class _Chunk:
        def __init__(self, c: str) -> None:
            self.content = c
            self.tool_call_chunks: list[Any] = []

    ev: dict[str, Any] = {
        "event": "on_chat_model_stream",
        "data": {"chunk": _Chunk(text)},
        "name": "ChatModel",
        "run_id": "r",
    }
    if tags is not None:
        ev["tags"] = tags
    return ev


@pytest.mark.asyncio
async def test_stream_filters_internal_tagged_token_events() -> None:
    """Tokens emitted with INTERNAL_TAG must not reach the SSE stream.

    Regression test for the extractor/compactor leak: without this filter,
    their JSON output was appearing in the user-facing chat bubble after
    the real agent reply finished.
    """
    from langgraph_kit.core.internal_tags import (
        INTERNAL_TAG,
        MEMORY_EXTRACTION_TAG,
    )

    events = [
        _token_event("Hello from agent"),
        # Extractor-tagged events that MUST be dropped:
        _token_event("[", tags=[INTERNAL_TAG, MEMORY_EXTRACTION_TAG]),
        _token_event('{"action":"create"}', tags=[INTERNAL_TAG, MEMORY_EXTRACTION_TAG]),
        _token_event("]", tags=[INTERNAL_TAG, MEMORY_EXTRACTION_TAG]),
    ]

    graph = _FakeGraph(events)
    chunks: list[str] = []
    async for chunk in stream_agent_events(graph, {}, {}):
        chunks.append(chunk)

    # Collect all emitted tokens from the SSE stream. Each chunk now
    # carries an SSE ``id:`` line plus a ``data:`` line — pluck the data
    # line specifically rather than naively stripping the "data: " prefix
    # off the whole chunk.
    tokens: list[str] = []
    for raw in chunks:
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "token" in payload:
                tokens.append(payload["token"])

    joined = "".join(tokens)
    # The main agent's reply must reach the stream.
    assert "Hello from agent" in joined
    # The extractor's JSON must NOT appear in the user-visible stream.
    assert '"action":"create"' not in joined
    assert "[" not in joined
    assert "]" not in joined


# ---------------------------------------------------------------------------
# Live graph overlay (issue #86) — node_entered / node_exited SSE events.
# ---------------------------------------------------------------------------


def _node_event(
    kind: str,
    node_name: str,
    *,
    run_id: str = "node-run",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a ``on_chain_start`` / ``on_chain_end`` event with the
    ``langgraph_node`` metadata key the streaming layer looks for."""
    ev: dict[str, Any] = {
        "event": kind,
        "data": {},
        "name": node_name,
        "run_id": run_id,
        "metadata": {"langgraph_node": node_name},
    }
    if tags is not None:
        ev["tags"] = tags
    return ev


def _extract_payloads(chunks: list[str]) -> list[dict[str, Any]]:
    """Pull JSON ``data:`` payloads out of SSE chunks (skip [DONE])."""
    out: list[dict[str, Any]] = []
    for raw in chunks:
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if not data or data == "[DONE]":
                continue
            try:
                out.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return out


@pytest.mark.asyncio
async def test_node_entered_and_exited_events_emitted_for_graph_nodes() -> None:
    """``on_chain_start`` / ``on_chain_end`` with langgraph_node fire SSE events."""
    events = [
        _node_event("on_chain_start", "alpha"),
        _node_event("on_chain_end", "alpha"),
        _node_event("on_chain_start", "beta"),
        _node_event("on_chain_end", "beta"),
    ]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered = [p["node_entered"]["name"] for p in payloads if "node_entered" in p]
    exited = [p["node_exited"]["name"] for p in payloads if "node_exited" in p]

    assert entered == ["alpha", "beta"]
    assert exited == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_node_events_skip_non_graph_chains() -> None:
    """Without ``langgraph_node`` in metadata the chain event is not surfaced.

    LangGraph fires ``on_chain_start`` for every Runnable in the call
    tree (middleware, langchain helpers, etc.). The overlay only cares
    about declared graph nodes.
    """
    raw_chain = {
        "event": "on_chain_start",
        "data": {},
        "name": "RunnablePassthrough",
        "run_id": "r-passthrough",
        "metadata": {},  # no langgraph_node key
    }
    events = [raw_chain, _node_event("on_chain_start", "alpha")]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered = [p["node_entered"]["name"] for p in payloads if "node_entered" in p]
    # Only ``alpha`` (the declared graph node) surfaces.
    assert entered == ["alpha"]


@pytest.mark.asyncio
async def test_internal_tagged_chain_events_are_dropped() -> None:
    """Memory-extraction / consolidation middleware mustn't surface as nodes."""
    from langgraph_kit.core.internal_tags import INTERNAL_TAG

    events = [
        _node_event("on_chain_start", "memory_extractor", tags=[INTERNAL_TAG]),
        _node_event("on_chain_start", "user_node"),
    ]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered = [p["node_entered"]["name"] for p in payloads if "node_entered" in p]
    # The internally-tagged event is dropped before the node-event branch
    # ever sees it; only the user-graph node fires.
    assert entered == ["user_node"]


@pytest.mark.asyncio
async def test_repeated_node_entered_for_same_node_is_coalesced() -> None:
    """Sub-channel fan-in fires multiple ``on_chain_start`` for one node;
    overlay only cares about the transition."""
    events = [
        _node_event("on_chain_start", "alpha", run_id="run-a-1"),
        _node_event("on_chain_start", "alpha", run_id="run-a-2"),
        _node_event("on_chain_start", "alpha", run_id="run-a-3"),
        _node_event("on_chain_end", "alpha"),
    ]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered_count = sum(1 for p in payloads if "node_entered" in p)
    # Only one ``node_entered`` despite three ``on_chain_start`` events.
    assert entered_count == 1


@pytest.mark.asyncio
async def test_node_entered_re_fires_after_exit() -> None:
    """Same node re-entering (e.g. loop) should fire ``node_entered`` again."""
    events = [
        _node_event("on_chain_start", "loop_node"),
        _node_event("on_chain_end", "loop_node"),
        _node_event("on_chain_start", "loop_node"),  # second iteration
        _node_event("on_chain_end", "loop_node"),
    ]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered_count = sum(1 for p in payloads if "node_entered" in p)
    assert entered_count == 2


@pytest.mark.asyncio
async def test_node_event_id_matches_run_id() -> None:
    """Event ``id`` exposes LangGraph's run_id so callers can correlate."""
    events = [_node_event("on_chain_start", "alpha", run_id="abc123")]
    chunks: list[str] = []
    async for chunk in stream_agent_events(_FakeGraph(events), {}, {}):
        chunks.append(chunk)

    payloads = _extract_payloads(chunks)
    entered = next(p["node_entered"] for p in payloads if "node_entered" in p)
    assert entered["id"] == "abc123"
    assert entered["name"] == "alpha"
