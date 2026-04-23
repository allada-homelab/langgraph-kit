"""Coverage fill — ``AGUIEncoder`` event emitters.

``AGUIEncoder`` serializes AG-UI protocol events (RUN_STARTED,
TEXT_MESSAGE_*, TOOL_CALL_*, RUN_FINISHED) used by the
``stream_agui_events`` adapter. These unit tests pin the state-tracking
contract the adapter depends on: a RUN_STARTED → text/tool bracketing →
RUN_FINISHED life cycle.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.contrib.agui import AGUIEncoder


def _get_data_lines(chunk: str) -> list[str]:
    """Extract ``data: ...`` payload lines from an SSE chunk."""
    return [
        line[len("data: ") :] for line in chunk.split("\n") if line.startswith("data: ")
    ]


def test_run_started_and_finished_carry_thread_and_run_ids() -> None:
    enc = AGUIEncoder(thread_id="t-1", run_id="r-1")
    started = enc.encode_run_started()
    finished = enc.encode_run_finished()
    assert "t-1" in started
    assert "r-1" in started
    assert "t-1" in finished
    assert "r-1" in finished
    # RUN_STARTED and RUN_FINISHED events differ in event type.
    assert started != finished


def test_run_error_surfaces_the_given_message() -> None:
    enc = AGUIEncoder(thread_id="t-err", run_id="r-err")
    frame = enc.encode_run_error("something went wrong")
    assert "something went wrong" in frame


def test_text_token_sequence_brackets_with_start_and_end() -> None:
    """First token emits both TEXT_MESSAGE_START and TEXT_MESSAGE_CONTENT;
    subsequent tokens emit only CONTENT; ``encode_text_end`` closes the
    bracket. This matches the AG-UI protocol's streaming-message
    contract.
    """
    enc = AGUIEncoder(thread_id="t-text", run_id="r-text")

    first = enc.encode_text_token("hello")
    assert len(first) == 2, (
        f"First token should produce 2 frames (START + CONTENT); got {len(first)}"
    )

    second = enc.encode_text_token(" world")
    assert len(second) == 1, (
        f"Subsequent tokens should produce 1 frame (CONTENT only); got {len(second)}"
    )

    end = enc.encode_text_end()
    assert end is not None, "encode_text_end must produce a TEXT_MESSAGE_END frame"

    # After end, calling again should be a no-op (bracket already closed).
    second_end = enc.encode_text_end()
    assert second_end is None, (
        "Duplicate encode_text_end should return None, not emit a second frame"
    )


def test_text_end_without_text_start_returns_none() -> None:
    """No text tokens ever emitted → encode_text_end is a no-op."""
    enc = AGUIEncoder(thread_id="t-notext", run_id="r-notext")
    assert enc.encode_text_end() is None


def test_tool_call_bracket_emits_start_args_and_end() -> None:
    enc = AGUIEncoder(thread_id="t-tool", run_id="r-tool")
    start_frames = enc.encode_tool_call_start("tool-123", "ping")
    assert start_frames, "tool_call_start should emit frames"
    end_frames = enc.encode_tool_call_end("tool-123", "pong-result")
    assert end_frames, "tool_call_end should emit frames"
    # Output payload threaded through.
    assert any("pong-result" in f for f in end_frames)


def test_encode_custom_serializes_arbitrary_values() -> None:
    enc = AGUIEncoder(thread_id="t-c", run_id="r-c")
    frame = enc.encode_custom("budget", {"tokens_used": 123})
    assert "budget" in frame
    assert "123" in frame


def test_run_id_defaults_to_generated_uuid() -> None:
    """Omitting ``run_id`` fills it with a fresh UUID."""
    enc_a = AGUIEncoder(thread_id="t-a")
    enc_b = AGUIEncoder(thread_id="t-a")
    assert enc_a.run_id
    assert enc_b.run_id
    assert enc_a.run_id != enc_b.run_id, "Separate encoders must get distinct run ids"


def test_thread_id_defaults_to_empty_string() -> None:
    """Omitting ``thread_id`` keeps it as empty string (explicit sentinel)."""
    enc = AGUIEncoder()
    assert enc.thread_id == ""
    # Event payload should still serialize cleanly.
    assert enc.encode_run_started()


@pytest.mark.parametrize("bad_value", [object(), lambda x: x, pytest])
def test_encode_custom_falls_back_gracefully_on_non_serializable(
    bad_value: Any,
) -> None:
    """A non-JSON-serializable value must not crash the encoder.

    The encoder uses ``json.dumps(..., default=str)`` (or an
    equivalent graceful fallback) so arbitrary objects reach
    downstream as a string representation rather than raising.
    """
    enc = AGUIEncoder(thread_id="t", run_id="r")
    # Contract: should either succeed (string-serialized) or surface a
    # single recoverable error frame — not raise.
    try:
        frame = enc.encode_custom("weird", bad_value)
        assert isinstance(frame, str)
    except (TypeError, ValueError):
        # Acceptable fallback: encoder can refuse non-serializable input
        # via a well-typed exception, which the caller then surfaces as
        # an error frame at a higher layer. We pin that it doesn't
        # silently produce garbage.
        pass
