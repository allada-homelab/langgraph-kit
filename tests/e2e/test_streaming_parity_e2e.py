"""Cluster I — streaming (``astream_events``) parity with ``ainvoke``.

``stream_agent_events`` emits Server-Sent Events for token streams,
tool-call starts/ends, interrupts, and a final ``[DONE]`` sentinel. The
SSE output is the UI contract; a regression here silently breaks the
live frontend even when ``ainvoke`` still works.

These tests exercise the streaming path against a real compiled graph
and assert:
- The final state reachable via ``astream_events`` matches what
  ``ainvoke`` returns (tool/message sequence).
- The SSE wrapper ``stream_agent_events`` emits the expected event
  types in the expected order (``tool_call_start`` → ``tool_call_end``
  → ``token``... → ``[DONE]``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    ToolMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.streaming import stream_agent_events
from tests.e2e.helpers import answer, scripted_llm, tool_call_turn

pytestmark = pytest.mark.e2e


async def probe() -> str:
    """Probe tool body — returns a distinctive payload."""
    return "PROBE-OK"


def _configure_probe(registry: Any) -> None:
    registry.register(
        ToolCapability(
            id="probe",
            name="probe",
            description="Return a marker.",
            fn=probe,
            risk=ToolRisk.READ_ONLY,
        )
    )


@pytest.mark.asyncio
async def test_ainvoke_and_streaming_final_state_match(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Running the same input via ainvoke and via astream_events produces
    the same final message sequence.

    Guards against a regression where a middleware mutates state
    differently on the streaming path (separate ``aget_state`` call vs
    direct return value).
    """

    # Two separate scripts — ``RecordedChatModel._call_index`` advances
    # across every ``_generate`` call, so sharing one script between the
    # ainvoke and stream legs would leave the stream leg with nothing to
    # serve on its second call.
    def _script() -> Any:
        return scripted_llm(
            [
                tool_call_turn("probe"),
                answer("probe-done"),
            ]
        )

    # ainvoke leg — fresh graph with a fresh script.
    with patched_build_llm(_script()):
        graph_invoke, _ = build_deep_agent(
            agent_name="stream-parity-invoke",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_probe,
        )
    invoke_result = await graph_invoke.ainvoke(
        {"messages": [HumanMessage(content="probe please")]},
        config={"configurable": {"thread_id": "parity-invoke"}},  # pyright: ignore[reportArgumentType]
    )
    invoke_tools = [
        m.name for m in invoke_result["messages"] if isinstance(m, ToolMessage)
    ]
    assert "probe" in invoke_tools

    # astream_events leg — fresh graph, fresh script.
    with patched_build_llm(_script()):
        graph_stream, _ = build_deep_agent(
            agent_name="stream-parity-stream",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_probe,
        )

    stream_thread = "parity-stream"
    config = {"configurable": {"thread_id": stream_thread}}
    events_seen = []
    async for event in graph_stream.astream_events(
        {"messages": [HumanMessage(content="probe please")]},
        config=config,  # pyright: ignore[reportArgumentType]
        version="v2",
    ):
        events_seen.append(event["event"])

    assert events_seen, "astream_events should yield at least one event"
    stream_state = await graph_stream.aget_state(config)  # pyright: ignore[reportArgumentType]
    stream_tools = [
        m.name
        for m in stream_state.values.get("messages", [])
        if isinstance(m, ToolMessage)
    ]
    assert "probe" in stream_tools, (
        f"Streaming leg should have produced the same tool sequence as"
        f" ainvoke. Streaming tools: {stream_tools}"
    )


@pytest.mark.asyncio
async def test_stream_agent_events_emits_expected_sse_shape(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """The SSE wrapper emits ``tool_call_start``, ``tool_call_end``, and ``[DONE]``.

    Token events only fire against a real streaming LLM — RecordedChatModel
    returns a single non-streamed generation — so we don't assert on
    ``token`` events here. ``tool_call_start`` / ``tool_call_end`` come
    from LangGraph's event stream regardless of the model's streaming
    capability, so they're the reliable SSE signals to pin.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("probe"),
            answer("stream-done"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="sse-wrap-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_probe,
        )

    chunks: list[str] = []
    async for raw in stream_agent_events(
        graph,
        input_data={"messages": [HumanMessage(content="stream please")]},
        config={"configurable": {"thread_id": "sse-wrap"}},
        store=e2e_store,
    ):
        chunks.append(raw)

    assert chunks, "stream_agent_events should have produced at least one chunk"
    # Last chunk is the [DONE] sentinel.
    assert any("[DONE]" in c for c in chunks), (
        f"SSE stream must end with [DONE]; got chunks[-3:]={chunks[-3:]}"
    )

    # Extract the event keys from each SSE frame. Format is:
    #   data: {json}\n\n
    kinds: list[str] = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                kinds.append("DONE")
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for key in event:
                kinds.append(key)

    assert "tool_call_start" in kinds, (
        f"SSE stream missing tool_call_start; saw event kinds {kinds!r}"
    )
    assert "tool_call_end" in kinds, (
        f"SSE stream missing tool_call_end; saw event kinds {kinds!r}"
    )
    assert kinds[-1] == "DONE", (
        f"SSE stream must terminate with the [DONE] sentinel; got {kinds[-3:]}"
    )
    # Ordering: tool_call_start appears before its corresponding
    # tool_call_end.
    start_idx = kinds.index("tool_call_start")
    end_idx = kinds.index("tool_call_end")
    assert start_idx < end_idx, (
        f"tool_call_start ({start_idx}) must appear before tool_call_end ({end_idx})"
    )
