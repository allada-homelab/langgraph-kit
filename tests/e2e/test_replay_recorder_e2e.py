"""Cluster I — ``ConversationRecorder`` record → replay round-trip.

The recorder attaches as a LangChain ``AsyncCallbackHandler`` via
``config["callbacks"]`` and captures every LLM + tool interaction the
run produces. The ``RecordedChatModel`` then replays those interactions
for deterministic tests.

The round-trip this file asserts: run a graph with recording enabled,
save/load the JSON serialization, inspect the captured interactions,
and then use the recording to drive a second run that reproduces the
tool sequence. Previously the recorder had only unit coverage of its
individual callback methods; the end-to-end contract — "a real graph
produces a valid recording that can actually drive a replay" — was not
tested.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.replay import ConversationRecorder, ConversationRecording
from langgraph_kit.replay.assertions import ReplayAssertions, assert_tool_sequence
from tests.e2e.helpers import (
    answer,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


async def ping() -> str:
    """Ping tool body — returns a marker the recording can identify.

    Named ``ping`` (not ``_ping``) because LangChain's StructuredTool
    derives the LLM-facing tool name from ``fn.__name__``, not from
    :attr:`ToolCapability.name`.
    """
    return "PONG-FROM-PING"


def _configure_ping(registry: Any) -> None:
    registry.register(
        ToolCapability(
            id="ping",
            name="ping",
            description="Respond with PONG.",
            fn=ping,
            risk=ToolRisk.READ_ONLY,
        )
    )


@pytest.mark.asyncio
async def test_recorder_captures_tool_sequence_from_real_graph_run(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Attach a recorder to a real graph; verify captured interactions."""
    scripted = scripted_llm(
        [
            tool_call_turn("ping"),
            answer("roundtrip done"),
        ]
    )
    recorder = ConversationRecorder(agent_id="record-e2e", thread_id="rec-1")

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="record-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_ping,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="ping it")]},
        config={  # pyright: ignore[reportArgumentType]
            "configurable": {"thread_id": "rec-1"},
            "callbacks": [recorder],
        },
    )

    recording = recorder.get_recording()
    # Recorder should have seen both the LLM turn and the tool call.
    assert len(recording.llm_interactions) >= 2, (
        f"Expected ≥2 LLM interactions; got {len(recording.llm_interactions)}"
    )
    assert "ping" in recording.tool_sequence, (
        f"Recorded tool_sequence should include 'ping'; got {recording.tool_sequence}"
    )
    # Output from the tool should carry our marker.
    ping_calls = [t for t in recording.tool_interactions if t.tool_name == "ping"]
    assert ping_calls, "ping tool interaction should have been recorded"
    assert "PONG-FROM-PING" in ping_calls[0].tool_output, (
        f"Tool output not captured; got {ping_calls[0].tool_output!r}"
    )


@pytest.mark.asyncio
async def test_recorder_save_load_roundtrip_preserves_interactions(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
    tmp_path: Path,
) -> None:
    """``recorder.save(path)`` + ``ConversationRecorder.load(path)`` round-trips.

    JSON schema drift would silently break replay-driven CI fixtures.
    This test pins the serialization contract.
    """
    scripted = scripted_llm([answer("recorded")])
    recorder = ConversationRecorder(agent_id="rt-agent", thread_id="rt-thread")

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="rt-agent",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="capture me")]},
        config={  # pyright: ignore[reportArgumentType]
            "configurable": {"thread_id": "rt-thread"},
            "callbacks": [recorder],
        },
    )

    path = tmp_path / "recording.json"
    recorder.save(path)
    assert path.exists(), "recorder.save should have written the JSON"

    # Re-parse and compare structural invariants.
    raw = json.loads(path.read_text())
    assert raw["agent_id"] == "rt-agent"
    assert raw["thread_id"] == "rt-thread"

    loaded = ConversationRecorder.load(path)
    assert isinstance(loaded, ConversationRecording)
    assert loaded.agent_id == "rt-agent"
    # Each interaction preserved its type tag across the round-trip.
    assert all(i.kind in ("llm", "tool") for i in loaded.interactions), (
        f"Interactions should all be llm/tool; got {[i.kind for i in loaded.interactions]}"
    )


@pytest.mark.asyncio
async def test_replay_assertions_detect_mismatches(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """ReplayAssertions raises when the replayed recording diverges from the original."""
    scripted_original = scripted_llm(
        [
            tool_call_turn("ping"),
            answer("original"),
        ]
    )
    rec_original = ConversationRecorder(agent_id="a1", thread_id="t1")
    with patched_build_llm(scripted_original):
        graph, _ = build_deep_agent(
            agent_name="a1",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_ping,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="ping")]},
        config={  # pyright: ignore[reportArgumentType]
            "configurable": {"thread_id": "t1"},
            "callbacks": [rec_original],
        },
    )
    original = rec_original.get_recording()

    # "Replay" a DIFFERENT run that skipped the tool call.
    scripted_replay = scripted_llm([answer("replay-skip")])
    rec_replay = ConversationRecorder(agent_id="a2", thread_id="t2")
    with patched_build_llm(scripted_replay):
        graph2, _ = build_deep_agent(
            agent_name="a2",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    await graph2.ainvoke(
        {"messages": [HumanMessage(content="skip")]},
        config={  # pyright: ignore[reportArgumentType]
            "configurable": {"thread_id": "t2"},
            "callbacks": [rec_replay],
        },
    )
    replayed = rec_replay.get_recording()

    assertions = ReplayAssertions(original, replayed)
    # Same tool sequence would be false: original called ping, replay didn't.
    with pytest.raises(AssertionError, match=r"[Tt]ool sequence"):
        assertions.assert_same_tool_sequence()

    # assert_tool_sequence standalone helper: original sequence ["ping"]
    # matches itself but mismatches the replay's empty sequence.
    assert_tool_sequence(original, ["ping"])
    with pytest.raises(AssertionError, match=r"[Tt]ool sequence"):
        assert_tool_sequence(replayed, ["ping"])

    # assert_tool_called on replayed (which has zero tool calls) for an
    # existing name should raise.
    with pytest.raises(AssertionError, match="never called"):
        assertions.assert_tool_called("ping")

    # assert_final_output_contains passes when the substring matches.
    assertions.assert_final_output_contains("replay-skip")
    with pytest.raises(AssertionError, match="does not contain"):
        assertions.assert_final_output_contains("totally-not-there")


@pytest.mark.asyncio
async def test_recording_drives_a_second_graph_run_deterministically(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A recording captured from run #1 can drive run #2 via RecordedChatModel.

    This is the end-to-end replay loop the kit's deterministic-test story
    depends on. If the output_message schema or tool-call preservation
    ever drifts, run #2 fails with a ReplayMismatchError.
    """
    from langgraph_kit.replay import RecordedChatModel

    scripted = scripted_llm(
        [
            tool_call_turn("ping"),
            answer("final"),
        ]
    )
    recorder = ConversationRecorder(agent_id="loop", thread_id="loop-1")
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="loop",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_ping,
        )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="ping")]},
        config={  # pyright: ignore[reportArgumentType]
            "configurable": {"thread_id": "loop-1"},
            "callbacks": [recorder],
        },
    )
    recording = recorder.get_recording()

    # Now use the captured recording to drive a fresh graph.
    replayed = RecordedChatModel(recording=recording)
    with patched_build_llm(replayed):
        graph2, _ = build_deep_agent(
            agent_name="loop-2",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure_ping,
        )
    result = await graph2.ainvoke(
        {"messages": [HumanMessage(content="ping again")]},
        config={"configurable": {"thread_id": "loop-2"}},  # pyright: ignore[reportArgumentType]
    )
    # Tool was re-invoked through replay.
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        ToolMessage,
    )

    tool_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "ping"
    ]
    assert tool_msgs, "Replayed run should still invoke the ping tool"
