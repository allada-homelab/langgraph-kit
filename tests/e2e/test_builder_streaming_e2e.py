"""Cluster H — ``build_deep_agent`` exercised via streaming invocation.

Every other Cluster H e2e drives the graph through ``ainvoke``. This
one tops up the coverage by running the same graph through
``astream_events`` + the SSE wrapper, confirming the builder's output
is usable on both the request/response and streaming paths.

Distinct from ``test_streaming_parity_e2e.py`` (which asserts the
ainvoke ↔ astream final states match). This test's job is narrower:
confirm a default ``build_deep_agent`` is consumable by a streaming
caller without any extra configuration.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.streaming import stream_agent_events
from tests.e2e.helpers import answer, scripted_llm, tool_call_turn

pytestmark = pytest.mark.e2e


async def heartbeat() -> str:
    """Tool that returns a marker distinct enough to search for in the SSE stream."""
    return "HEARTBEAT-OK-3f91"


def _configure_heartbeat(registry: Any) -> None:
    registry.register(
        ToolCapability(
            id="heartbeat",
            name="heartbeat",
            description="Stream-mode test tool.",
            fn=heartbeat,
            risk=ToolRisk.READ_ONLY,
        )
    )


@pytest.mark.asyncio
async def test_build_deep_agent_graph_drives_sse_stream(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A default ``build_deep_agent`` run via ``stream_agent_events`` emits valid SSE.

    Verifies the happy-path SSE contract against a builder output that
    hasn't been set up for streaming specifically:
    - ``tool_call_start`` and ``tool_call_end`` appear with matching
      ``id`` and ``name`` fields.
    - The tool output (``heartbeat-ok-3f91``) reaches the stream as the
      ``tool_call_end.output``.
    - The stream ends with the ``[DONE]`` sentinel.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("heartbeat"),
            answer("alive"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="builder-stream-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_heartbeat,
        )

    events: list[dict[str, Any]] = []
    done_seen = False
    async for raw in stream_agent_events(
        graph,
        input_data={"messages": [HumanMessage(content="heartbeat please")]},
        config={"configurable": {"thread_id": "builder-stream"}},
        store=e2e_store,
    ):
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                done_seen = True
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pytest.fail(
                    f"stream_agent_events emitted a non-JSON data frame: {payload!r}"
                )

    assert done_seen, "SSE stream must terminate with the [DONE] sentinel"

    starts = [e["tool_call_start"] for e in events if "tool_call_start" in e]
    ends = [e["tool_call_end"] for e in events if "tool_call_end" in e]
    assert starts, "Expected at least one tool_call_start event"
    assert ends, "Expected at least one tool_call_end event"

    # heartbeat-named events appear in both streams.
    heartbeat_starts = [s for s in starts if s.get("name") == "heartbeat"]
    heartbeat_ends = [e for e in ends if e.get("name") == "heartbeat"]
    assert heartbeat_starts, f"No tool_call_start for heartbeat; starts={starts!r}"
    assert heartbeat_ends, f"No tool_call_end for heartbeat; ends={ends!r}"

    # Start and end share a run id (id in SSE == run_id from LangGraph).
    assert heartbeat_starts[0].get("id") == heartbeat_ends[0].get("id"), (
        f"tool_call_start and tool_call_end must carry matching run ids;"
        f" start={heartbeat_starts[0]!r}, end={heartbeat_ends[0]!r}"
    )

    assert "HEARTBEAT-OK-3f91" in str(heartbeat_ends[0].get("output", "")), (
        f"tool_call_end.output should carry the tool's return value;"
        f" got {heartbeat_ends[0]!r}"
    )
