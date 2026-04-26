"""Conversation replay testing framework.

Record real agent conversations, replay them with mocked LLM responses,
and assert on tool call sequences and outputs for deterministic regression testing.

Quick start::

    from langgraph_kit.replay import ConversationRecorder, ReplayRunner

    # Record (run once with real LLM)
    recorder = ConversationRecorder("my-agent", "thread-1")
    config["callbacks"].append(recorder)
    await graph.ainvoke(input_data, config=config)
    recorder.save(Path("tests/fixtures/my_flow.json"))

    # Replay (deterministic in CI)
    runner = ReplayRunner(Path("tests/fixtures/my_flow.json"), build_my_agent)
    assertions = await runner.run_and_assert()
    assertions.assert_same_tool_sequence()
"""

from langgraph_kit.replay.assertions import (
    ReplayAssertions,
    assert_replay_matches,
    assert_tool_sequence,
)
from langgraph_kit.replay.models import (
    ConversationRecording,
    LLMInteraction,
    RecordingOverrides,
    ToolInteraction,
)
from langgraph_kit.replay.player import RecordedChatModel, ReplayMismatchError
from langgraph_kit.replay.recorder import ConversationRecorder
from langgraph_kit.replay.runner import ReplayRunner

__all__ = [
    "ConversationRecorder",
    "ConversationRecording",
    "LLMInteraction",
    "RecordedChatModel",
    "RecordingOverrides",
    "ReplayAssertions",
    "ReplayMismatchError",
    "ReplayRunner",
    "ToolInteraction",
    "assert_replay_matches",
    "assert_tool_sequence",
]
