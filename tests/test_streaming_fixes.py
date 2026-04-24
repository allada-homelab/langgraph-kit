"""Regression tests for Phase F streaming-layer fixes."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgraph_kit.streaming import _TRAILING_ARTIFACT_RE, stream_agent_events


def test_trailing_artifact_regex_fires_only_on_fenced_form() -> None:
    # Fenced shapes that models (Qwen via vLLM) leak at end-of-stream.
    assert _TRAILING_ARTIFACT_RE.search("hello\n\n```json\n[]\n```")
    assert _TRAILING_ARTIFACT_RE.search("hello\n```\n[]\n```")

    # Bare trailing ``[]`` is legitimate content ("returned `[]`") and must
    # NOT be stripped.
    assert _TRAILING_ARTIFACT_RE.search("the function returned []") is None
    assert _TRAILING_ARTIFACT_RE.search("[]") is None
    assert _TRAILING_ARTIFACT_RE.search("see [1, 2, 3]") is None


class _RaisingGraph:
    async def astream_events(
        self, input_data: Any, config: Any = None, version: str = "v2"
    ) -> Any:
        # Produce one event, then raise.
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("hi")}, "tags": ()}
        raise RuntimeError("upstream boom")

    async def aget_state(self, config: Any) -> Any:
        return MagicMock(values={}, tasks=[])

    config: Any = None


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_call_chunks: list[Any] = []


@pytest.mark.asyncio
async def test_stream_emits_error_event_and_done_on_failure() -> None:
    graph = _RaisingGraph()
    frames: list[str] = []
    async for frame in stream_agent_events(
        graph, {"messages": []}, {"configurable": {"thread_id": "t"}}
    ):
        frames.append(frame)

    # Expect an error frame AND a DONE frame, in that order.
    error_idx = next(
        (i for i, f in enumerate(frames) if '"error"' in f), None
    )
    done_idx = next(
        (i for i, f in enumerate(frames) if "[DONE]" in f), None
    )
    assert error_idx is not None, f"No error frame in {frames!r}"
    assert done_idx is not None, f"No [DONE] frame in {frames!r}"
    assert error_idx < done_idx, "error must precede [DONE]"

    # Error frame carries a readable exception type + message.
    error_payload = json.loads(frames[error_idx].removeprefix("data: ").strip())
    assert "error" in error_payload
    assert "RuntimeError" in error_payload["error"]["message"]
    assert "upstream boom" in error_payload["error"]["message"]


class _ToolOnlyGraph:
    """Graph whose state has a final AIMessage without COMMAND_RESULT_MARKER."""

    async def astream_events(
        self, input_data: Any, config: Any = None, version: str = "v2"
    ) -> Any:
        if False:
            yield None  # pragma: no cover

    async def aget_state(self, config: Any) -> Any:
        # Simulate a tool-only run: last AI message with no command marker.
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
        )

        msg = AIMessage(content="tool-only finalizer")

        class _State:
            values: dict[str, Any] = {"messages": [msg]}  # noqa: RUF012
            tasks: list[Any] = []  # noqa: RUF012

        return _State()

    config: Any = None


@pytest.mark.asyncio
async def test_command_result_not_emitted_without_marker() -> None:
    """Tool-only runs must not get their last AIMessage mis-labelled as a command result."""
    graph = _ToolOnlyGraph()
    frames: list[str] = []
    async for frame in stream_agent_events(
        graph, {"messages": []}, {"configurable": {"thread_id": "t"}}
    ):
        frames.append(frame)

    assert not any('"command_result"' in f for f in frames), (
        "command_result emitted without the marker — regression."
    )


class _CommandGraph:
    """Graph whose state has a last AIMessage with the COMMAND_RESULT_MARKER set."""

    async def astream_events(
        self, input_data: Any, config: Any = None, version: str = "v2"
    ) -> Any:
        if False:
            yield None  # pragma: no cover

    async def aget_state(self, config: Any) -> Any:
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
        )

        from langgraph_kit.core.commands.middleware import COMMAND_RESULT_MARKER

        msg = AIMessage(
            content="/help output here",
            additional_kwargs={COMMAND_RESULT_MARKER: True},
        )

        class _State:
            values: dict[str, Any] = {"messages": [msg]}  # noqa: RUF012
            tasks: list[Any] = []  # noqa: RUF012

        return _State()

    config: Any = None


@pytest.mark.asyncio
async def test_command_result_emitted_when_marker_present() -> None:
    graph = _CommandGraph()
    frames: list[str] = []
    async for frame in stream_agent_events(
        graph, {"messages": []}, {"configurable": {"thread_id": "t"}}
    ):
        frames.append(frame)

    assert any('"command_result"' in f for f in frames), (
        "Expected command_result event when marker is set."
    )
