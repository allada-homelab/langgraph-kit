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

    # Collect all emitted tokens from the SSE stream.
    tokens: list[str] = []
    for raw in chunks:
        for line in raw.strip().split("\n\n"):
            line = line.removeprefix("data: ").strip()
            if not line or line == "[DONE]":
                continue
            try:
                payload = json.loads(line)
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
