"""Streaming: consume the kit's SSE event stream end-to-end.

What this shows
---------------
- Wiring up :func:`stream_agent_events` against a kit-built graph
- Parsing the SSE wire format (``id:`` line + ``data:`` line per chunk)
- Distinguishing the kit's event types (token / tool_call_start /
  heartbeat / [DONE]) from a real client's perspective

The kit emits SSE chunks with monotonically-increasing ``id`` lines so a
reconnecting client can resume via ``Last-Event-ID``. Heartbeats are
disabled here so the demo finishes quickly; production callers leave the
default 15s cadence so proxies don't drop idle streams.

How to run
----------
    uv run python -m examples.streaming_sse_events

Expected output (hermetic)
--------------------------
    user: How are you doing today?
    --- SSE chunks ---
      id=0  done='[DONE]'
    Total events: 1

(In hermetic mode the scripted LLM returns a single non-streamed
``AIMessage`` so no ``on_chat_model_stream`` events fire — the [DONE]
sentinel is the whole stream. Run in real-LLM mode to see live token
chunks.)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from examples._lib import (
    answer,
    assert_real_llm_or_skip,
    banner,
    configure_real_llm,
    hermetic,
    line,
    make_in_memory_persistence,
    patch_build_llm,
    scripted_llm,
    tmp_workspace,
)


def _parse_sse_chunk(chunk: str) -> tuple[str | None, Any]:
    """Pull the ``id`` and decoded ``data`` payload out of one SSE block."""
    seq: str | None = None
    payload: Any = None
    for raw_line in chunk.splitlines():
        if raw_line.startswith("id:"):
            seq = raw_line[3:].strip()
        elif raw_line.startswith("data:"):
            body = raw_line[5:].strip()
            if body == "[DONE]":
                payload = "[DONE]"
            else:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = body
    return seq, payload


async def main() -> None:
    banner("streaming_sse_events")
    user_message = "How are you doing today?"
    line(f"user: {user_message}")

    with tmp_workspace() as workspace:
        if hermetic():
            llm = scripted_llm([answer("I'm doing well, thanks!")])
            with patch_build_llm(llm):
                await _stream_one_turn(workspace, user_message)
        else:
            assert_real_llm_or_skip()
            configure_real_llm(workspace)
            await _stream_one_turn(workspace, user_message)


async def _stream_one_turn(workspace: object, user_message: str) -> None:
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit import stream_agent_events
    from langgraph_kit.graphs.echo_agent import build_graph

    _ = workspace

    checkpointer, store = make_in_memory_persistence()
    graph = build_graph(checkpointer, store)
    config = {"configurable": {"thread_id": "demo-stream"}}

    line("--- SSE chunks ---")
    count = 0
    async for chunk in stream_agent_events(
        graph,
        {"messages": [HumanMessage(content=user_message)]},
        config,
        store=store,
        heartbeat_interval=None,  # quiet the demo so output is deterministic
    ):
        seq, payload = _parse_sse_chunk(chunk)
        kind = "done" if payload == "[DONE]" else "token"
        line(f"  id={seq}  {kind}={payload!r}")
        count += 1

    line(f"Total events: {count}")


if __name__ == "__main__":
    asyncio.run(main())
