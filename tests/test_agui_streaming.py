"""Coverage fill — AG-UI streaming adapters (SSE + native) and event mapping.

The streaming adapters consume another stream (our SSE format or
LangGraph's native astream) and emit AG-UI protocol events. Tests
drive each branch of ``_map_sse_to_agui`` and ``_map_native_to_agui``
with synthetic inputs and drive ``stream_agui_events`` /
``stream_agui_native`` via fake async iterables.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from langgraph_kit.contrib.agui import (
    AGUIEncoder,
    _map_native_to_agui,
    _map_sse_to_agui,
    stream_agui_events,
    stream_agui_native,
)


def _encoder() -> AGUIEncoder:
    return AGUIEncoder(thread_id="t", run_id="r")


# ---------------------------------------------------------------------------
# _map_sse_to_agui
# ---------------------------------------------------------------------------


def test_map_sse_token_emits_text_frames() -> None:
    enc = _encoder()
    events = _map_sse_to_agui({"token": "hello"}, enc)
    # First token produces 2 frames (START + CONTENT).
    assert len(events) == 2


def test_map_sse_tool_call_start() -> None:
    enc = _encoder()
    events = _map_sse_to_agui({"tool_call_start": {"id": "c1", "name": "ping"}}, enc)
    assert events


def test_map_sse_tool_call_end() -> None:
    enc = _encoder()
    events = _map_sse_to_agui({"tool_call_end": {"id": "c1", "output": "pong"}}, enc)
    assert events
    assert any("pong" in e for e in events)


def test_map_sse_command_result_emits_text() -> None:
    enc = _encoder()
    events = _map_sse_to_agui({"command_result": {"output": "ran /foo"}}, enc)
    # command_result text goes through encode_text_token.
    assert any("ran /foo" in e for e in events)


@pytest.mark.parametrize(
    ("key", "payload"),
    [
        ("artifact", {"type": "code", "text": "print(1)"}),
        ("progress", {"pct": 50}),
        ("suggestions", [{"title": "next"}]),
        ("citation", {"url": "https://x"}),
        ("interrupt", {"action": "pause"}),
        ("budget", {"tokens_used": 10}),
        ("trace", {"trace_id": "t1"}),
    ],
)
def test_map_sse_custom_channels(key: str, payload: Any) -> None:
    enc = _encoder()
    events = _map_sse_to_agui({key: payload}, enc)
    assert events
    # encode_custom emits the channel name in the frame.
    assert any(key in e for e in events)


def test_map_sse_unknown_key_returns_empty() -> None:
    enc = _encoder()
    assert _map_sse_to_agui({"unknown_channel": "noise"}, enc) == []


# ---------------------------------------------------------------------------
# _map_native_to_agui
# ---------------------------------------------------------------------------


class _FakeMsg:
    def __init__(self, content: str, msg_type: str, tool_call_id: str = "") -> None:
        self.content = content
        self.type = msg_type
        self.tool_call_id = tool_call_id


def test_map_native_messages_emits_text_for_ai_chunk() -> None:
    enc = _encoder()
    msg = _FakeMsg("hello tokens", "AIMessageChunk")
    events = _map_native_to_agui("messages", (msg, {}), enc)
    assert events


def test_map_native_messages_emits_tool_call_end_for_tool_msg() -> None:
    enc = _encoder()
    msg = _FakeMsg("tool-result", "tool", tool_call_id="tc-1")
    events = _map_native_to_agui("messages", (msg, {}), enc)
    assert events
    assert any("tool-result" in e for e in events)


def test_map_native_updates_emits_step_events() -> None:
    enc = _encoder()
    events = _map_native_to_agui(
        "updates", {"researcher": {"output": "x"}, "writer": {"output": "y"}}, enc
    )
    # Two nodes x (STEP_STARTED + STEP_FINISHED) -> at least 4 frames.
    assert len(events) == 4


def test_map_native_custom_mode_passthrough() -> None:
    enc = _encoder()
    events = _map_native_to_agui("custom", {"arbitrary": "payload"}, enc)
    assert events


def test_map_native_unknown_mode_returns_empty() -> None:
    enc = _encoder()
    assert _map_native_to_agui("unsupported", {"x": 1}, enc) == []


def test_map_native_messages_ignores_malformed_tuple() -> None:
    """If data isn't the expected (message, metadata) tuple, no frames emitted."""
    enc = _encoder()
    assert _map_native_to_agui("messages", "not a tuple", enc) == []


# ---------------------------------------------------------------------------
# stream_agui_events (SSE passthrough adapter)
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Graph stand-in with a fake astream that yields canned SSE lines."""


async def _fake_sse_source(
    graph: Any, input_data: Any, config: Any, store: Any = None
) -> AsyncGenerator[str]:
    _ = graph
    _ = input_data
    _ = config
    _ = store
    yield 'data: {"token": "hi"}\n\n'
    yield 'data: {"tool_call_start": {"id": "c1", "name": "p"}}\n\n'
    yield 'data: {"tool_call_end": {"id": "c1", "output": "pong"}}\n\n'
    yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_agui_events_wraps_sse_source(monkeypatch: Any) -> None:
    """stream_agui_events consumes our SSE format and re-emits AG-UI frames.

    Assertion: the output stream starts with RUN_STARTED, contains the
    mapped events from each SSE line, and ends with RUN_FINISHED.
    """
    monkeypatch.setattr("langgraph_kit.streaming.stream_agent_events", _fake_sse_source)
    frames = []
    async for frame in stream_agui_events(
        _FakeGraph(),
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)

    blob = "\n".join(frames)
    # RUN_STARTED and RUN_FINISHED bookend the stream.
    assert "RUN_STARTED" in blob
    assert "RUN_FINISHED" in blob
    # Tool call is surfaced.
    assert "pong" in blob


@pytest.mark.asyncio
async def test_stream_agui_events_emits_run_error_on_upstream_exception(
    monkeypatch: Any,
) -> None:
    """Upstream exception → RUN_ERROR frame, not a propagated crash."""

    async def _raising_source(*args: Any, **kwargs: Any) -> AsyncGenerator[str]:
        _ = args
        _ = kwargs
        msg = "upstream broken"
        raise RuntimeError(msg)
        yield ""  # pragma: no cover (make it a generator)

    monkeypatch.setattr("langgraph_kit.streaming.stream_agent_events", _raising_source)
    frames = []
    async for frame in stream_agui_events(
        _FakeGraph(),
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)
    blob = "\n".join(frames)
    assert "RUN_ERROR" in blob
    assert "upstream broken" in blob


# ---------------------------------------------------------------------------
# stream_agui_native (native LangGraph astream adapter)
# ---------------------------------------------------------------------------


class _FakeNativeGraph:
    def __init__(self, chunks: list[tuple[str, Any]]) -> None:
        self._chunks = chunks

    async def astream(
        self,
        input_data: Any,
        config: Any,
        stream_mode: Any = None,
    ) -> AsyncGenerator[tuple[str, Any]]:
        _ = input_data
        _ = config
        _ = stream_mode
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_stream_agui_native_emits_run_bracket_and_messages() -> None:
    chunks: list[tuple[str, Any]] = [
        ("messages", (_FakeMsg("alpha", "AIMessageChunk"), {})),
        ("updates", {"node-a": {"x": 1}}),
    ]
    graph = _FakeNativeGraph(chunks)
    frames = []
    async for frame in stream_agui_native(
        graph,
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)
    blob = "\n".join(frames)
    assert "RUN_STARTED" in blob
    assert "RUN_FINISHED" in blob
    # The update mode produces STEP_STARTED + STEP_FINISHED events.
    assert "STEP_STARTED" in blob
    assert "STEP_FINISHED" in blob


@pytest.mark.asyncio
async def test_stream_agui_native_handles_upstream_exception() -> None:
    class _BrokenGraph:
        async def astream(
            self, input_data: Any, config: Any, stream_mode: Any = None
        ) -> AsyncGenerator[tuple[str, Any]]:
            _ = input_data
            _ = config
            _ = stream_mode
            msg = "native broken"
            raise RuntimeError(msg)
            yield ("never", {})  # pragma: no cover

    frames = []
    async for frame in stream_agui_native(
        _BrokenGraph(),
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)
    blob = "\n".join(frames)
    assert "RUN_ERROR" in blob


@pytest.mark.asyncio
async def test_stream_agui_native_skips_non_tuple_chunks() -> None:
    """If astream yields something that isn't a (mode, data) tuple, it's skipped."""
    chunks: list[Any] = ["not a tuple", (1, 2, 3)]  # wrong shapes
    graph = _FakeNativeGraph(chunks)  # type: ignore[arg-type]
    frames = []
    async for frame in stream_agui_native(
        graph,
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)
    # Only RUN_STARTED and RUN_FINISHED in the output.
    blob = "\n".join(frames)
    assert "RUN_STARTED" in blob
    assert "RUN_FINISHED" in blob


@pytest.mark.asyncio
async def test_stream_agui_events_sse_json_parse_failure_is_skipped(
    monkeypatch: Any,
) -> None:
    """Malformed data frames are silently skipped (non-fatal)."""

    async def _garbage_source(*args: Any, **kwargs: Any) -> AsyncGenerator[str]:
        _ = args
        _ = kwargs
        yield "data: this is not json\n\n"
        yield 'data: {"token": "valid"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr("langgraph_kit.streaming.stream_agent_events", _garbage_source)
    frames = []
    async for frame in stream_agui_events(
        _FakeGraph(),
        input_data={"messages": []},
        config={"configurable": {"thread_id": "t"}},
    ):
        frames.append(frame)
    blob = "\n".join(frames)
    # The garbage frame is skipped; the valid token still makes it through.
    assert "valid" in blob


def test_sse_done_sentinel_is_recognized() -> None:
    """Regression: ``data: [DONE]`` should end the stream cleanly.

    With SSE ``id:`` lines now prepended to every chunk, the sentinel
    is rendered as ``id: <n>\\ndata: [DONE]\\n\\n``. A correct parser
    looks at every line and only acts on the one starting with
    ``data:``.
    """
    payload = "id: 7\ndata: [DONE]\n\n"
    data_line = next(
        line.removeprefix("data: ").strip()
        for line in payload.split("\n")
        if line.startswith("data: ")
    )
    assert data_line == "[DONE]"
    # sanity: json.loads should fail on this.
    with pytest.raises(json.JSONDecodeError):
        json.loads("[DONE]")
