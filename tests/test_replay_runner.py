"""Coverage fill — ``ReplayRunner`` + ``_extract_user_turns``.

``ReplayRunner`` loads a recording from disk, builds a graph with the
caller's builder injected a ``RecordedChatModel``, drives the graph
over each recorded user turn, and returns a replayed
``ConversationRecording``. The existing suite exercises the underlying
``RecordedChatModel`` and ``ConversationRecorder`` directly but never
runs the orchestrating ``ReplayRunner`` — this fills that gap.
"""

# NOTE: intentionally does NOT use ``from __future__ import annotations`` —
# LangGraph evaluates the StateGraph TypedDict's ``Annotated`` field at
# runtime via ``typing.get_type_hints()``, which needs the annotations
# to be live objects, not strings. See ``echo_agent.py`` for the same
# constraint in the kit.
import json
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)
from langchain_core.runnables import (  # pyright: ignore[reportMissingModuleSource]
    RunnableConfig,
)
from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
    InMemorySaver,
)
from langgraph.graph import (  # pyright: ignore[reportMissingModuleSource]
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import (  # pyright: ignore[reportMissingModuleSource]
    add_messages,
)
from typing_extensions import TypedDict

from langgraph_kit.replay.assertions import ReplayAssertions
from langgraph_kit.replay.models import (
    ConversationRecording,
    LLMInteraction,
    ToolInteraction,
)
from langgraph_kit.replay.runner import ReplayRunner, _extract_user_turns
from tests.conftest import MockStore


class _ReplayState(TypedDict):
    messages: Annotated[list[Any], add_messages]


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    """Write a small deterministic recording to disk for the runner to load."""
    recording = ConversationRecording(
        id=str(uuid4()),
        agent_id="runner-test",
        thread_id="runner-thread",
        created_at="2026-04-24T00:00:00Z",
        user_messages=[{"role": "user", "content": "ping from fixture"}],
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={"content": "pong", "tool_calls": []},
            ),
        ],
    )
    path = tmp_path / "recording.json"
    path.write_text(recording.model_dump_json())
    return path


def _simple_graph_builder(checkpointer: Any, store: Any, *, llm: Any) -> Any:
    """Minimal LangGraph StateGraph that forwards the LLM's response."""

    async def llm_node(state: dict, config: RunnableConfig) -> dict:
        response = await llm.ainvoke(state["messages"], config=config)
        return {"messages": [response]}

    builder = StateGraph(_ReplayState)
    builder.add_node("llm", llm_node)  # pyright: ignore[reportArgumentType]
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    _ = store  # unused in this minimal graph
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_replay_runner_runs_and_captures_turns(
    recording_path: Path,
) -> None:
    runner = ReplayRunner(
        recording_path=recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run()

    # The replay ran a single turn and emitted one LLM interaction.
    assert isinstance(replayed, ConversationRecording)
    assert len(replayed.llm_interactions) >= 1


@pytest.mark.asyncio
async def test_replay_runner_run_and_assert_returns_assertions(
    recording_path: Path,
) -> None:
    runner = ReplayRunner(
        recording_path=recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    # No tools in the fixture → check_tool_args=False avoids the tool-call-count
    # assertion which would otherwise be vacuous.
    assertions = await runner.run_and_assert(check_tool_args=False)
    assert isinstance(assertions, ReplayAssertions)
    assertions.assert_same_tool_sequence()  # Both empty sequences match.


def test_extract_user_turns_prefers_user_messages_field() -> None:
    """When ``user_messages`` is populated, extraction uses it directly."""
    rec = ConversationRecording(
        user_messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "first"},  # duplicate should be dropped
        ],
        interactions=[],
    )
    assert _extract_user_turns(rec) == ["first", "second"]


def test_extract_user_turns_falls_back_to_llm_input_messages() -> None:
    """With no ``user_messages`` field, extraction walks llm_interactions."""
    rec = ConversationRecording(
        user_messages=[],
        interactions=[
            LLMInteraction(
                sequence_num=1,
                input_messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
                output_message={"content": "hi"},
            ),
            LLMInteraction(
                sequence_num=2,
                input_messages=[
                    {"role": "user", "content": "follow-up"},
                ],
                output_message={"content": "ack"},
            ),
        ],
    )
    assert _extract_user_turns(rec) == ["hello", "follow-up"]


def test_extract_user_turns_dedupes_consecutive_duplicates() -> None:
    rec = ConversationRecording(
        user_messages=[],
        interactions=[
            LLMInteraction(
                sequence_num=1,
                input_messages=[{"role": "user", "content": "same"}],
                output_message={"content": "1"},
            ),
            LLMInteraction(
                sequence_num=2,
                input_messages=[{"role": "user", "content": "same"}],
                output_message={"content": "2"},
            ),
        ],
    )
    assert _extract_user_turns(rec) == ["same"]


def test_replay_runner_lazy_loads_original_recording(
    recording_path: Path,
) -> None:
    runner = ReplayRunner(
        recording_path=recording_path,
        graph_builder=_simple_graph_builder,
    )
    # First access triggers the read.
    first = runner.original
    # Second access returns the cached value (same object).
    second = runner.original
    assert first is second
    assert first.agent_id == "runner-test"


# ---------------------------------------------------------------------------
# ReplayAssertions — additional coverage
# ---------------------------------------------------------------------------


def test_assertions_output_similarity_exact_match_passes() -> None:
    rec = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={"content": "identical content"},
            ),
        ],
    )
    assertions = ReplayAssertions(rec, rec)
    assertions.assert_output_similarity(min_ratio=0.99)


def test_assertions_output_similarity_mismatch_raises() -> None:
    orig = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={"content": "alpha bravo charlie delta"},
            ),
        ],
    )
    replay = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={"content": "wildly different zebra xylophone"},
            ),
        ],
    )
    with pytest.raises(AssertionError, match="below threshold"):
        ReplayAssertions(orig, replay).assert_output_similarity(min_ratio=0.5)


def test_assertions_output_similarity_both_empty_passes() -> None:
    """Empty-vs-empty comparison is a vacuous match, not a failure."""
    empty = ConversationRecording(interactions=[])
    # No LLM interactions — _get_final_output raises, so the method guards.
    rec_with_empty_content = ConversationRecording(
        interactions=[LLMInteraction(sequence_num=1, output_message={"content": ""})]
    )
    ReplayAssertions(
        rec_with_empty_content, rec_with_empty_content
    ).assert_output_similarity(min_ratio=0.99)
    _ = empty  # keep reference for readability


def test_assertions_tool_not_called_raises_when_it_was() -> None:
    orig = ConversationRecording(
        interactions=[
            ToolInteraction(
                sequence_num=1,
                tool_name="should_not_run",
                tool_input={},
                tool_output="oops",
            ),
        ],
    )
    with pytest.raises(AssertionError, match="was called"):
        ReplayAssertions(orig, orig).assert_tool_not_called("should_not_run")


def test_assertions_same_tool_calls_flags_arg_mismatch() -> None:
    t_orig = ToolInteraction(
        sequence_num=1,
        tool_name="x",
        tool_input={"a": 1},
        tool_output="ok",
    )
    t_replay = ToolInteraction(
        sequence_num=1,
        tool_name="x",
        tool_input={"a": 2},  # different args
        tool_output="ok",
    )
    orig = ConversationRecording(interactions=[t_orig])
    replay = ConversationRecording(interactions=[t_replay])
    with pytest.raises(AssertionError, match="args mismatch"):
        ReplayAssertions(orig, replay).assert_same_tool_calls()


def test_assertions_same_tool_calls_flags_count_mismatch() -> None:
    orig = ConversationRecording(
        interactions=[
            ToolInteraction(sequence_num=1, tool_name="x", tool_output="a"),
            ToolInteraction(sequence_num=2, tool_name="x", tool_output="b"),
        ]
    )
    replay = ConversationRecording(
        interactions=[
            ToolInteraction(sequence_num=1, tool_name="x", tool_output="a"),
        ]
    )
    with pytest.raises(AssertionError, match="count mismatch"):
        ReplayAssertions(orig, replay).assert_same_tool_calls()


def test_assertions_final_output_matches_pattern() -> None:
    rec = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={"content": "answer: 42"},
            ),
        ],
    )
    ReplayAssertions(rec, rec).assert_final_output_matches(r"answer:\s*\d+")
    with pytest.raises(AssertionError, match="does not match"):
        ReplayAssertions(rec, rec).assert_final_output_matches(r"^no-match$")


def test_assertions_no_errors_passes_on_clean_recording() -> None:
    rec = ConversationRecording(
        interactions=[
            ToolInteraction(
                sequence_num=1,
                tool_name="ok",
                tool_output="fine",
                status="success",
            ),
        ]
    )
    ReplayAssertions(rec, rec).assert_no_errors()


def test_assertions_no_errors_raises_on_failing_tool() -> None:
    rec = ConversationRecording(
        interactions=[
            ToolInteraction(
                sequence_num=1,
                tool_name="broken",
                tool_output="traceback...",
                status="error",
            ),
        ]
    )
    with pytest.raises(AssertionError, match="error"):
        ReplayAssertions(rec, rec).assert_no_errors()


def test_assertions_handles_multipart_final_output() -> None:
    """Final LLM output can be a list of parts (multi-modal content)."""
    rec = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={
                    "content": [
                        {"type": "text", "text": "part-one"},
                        "bare-string",
                    ]
                },
            ),
        ],
    )
    assertions = ReplayAssertions(rec, rec)
    # _get_final_output merges list parts — both substrings must appear.
    assertions.assert_final_output_contains("part-one")
    assertions.assert_final_output_contains("bare-string")


def test_assertions_final_output_raises_on_no_llm_interactions() -> None:
    rec = ConversationRecording(interactions=[])
    with pytest.raises(AssertionError, match="No LLM interactions"):
        ReplayAssertions(rec, rec).assert_final_output_contains("anything")


def _make_user_aimessage_pair(human: str, ai: str) -> list[Any]:
    return [HumanMessage(content=human), AIMessage(content=ai)]


def test_extract_user_turns_empty_recording_returns_empty_list() -> None:
    assert _extract_user_turns(ConversationRecording(interactions=[])) == []


def test_recording_json_round_trip(recording_path: Path) -> None:
    """Loading an on-disk recording preserves the shape the runner expects."""
    raw = json.loads(recording_path.read_text())
    assert raw["agent_id"] == "runner-test"
    assert "interactions" in raw
    assert raw["interactions"][0]["kind"] == "llm"
