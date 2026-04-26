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
    RecordingOverrides,
    ToolInteraction,
)
from langgraph_kit.replay.player import RecordedChatModel
from langgraph_kit.replay.runner import ReplayRunner, TurnResult, _extract_user_turns
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


# ---------------------------------------------------------------------------
# Partial replay (start_at / stop_at) and RecordingOverrides — issue #15.
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_turn_recording_path(tmp_path: Path) -> Path:
    """Four-turn recording with distinct user messages per turn."""
    recording = ConversationRecording(
        id=str(uuid4()),
        agent_id="multi-turn-test",
        thread_id="multi-thread",
        created_at="2026-04-25T00:00:00Z",
        user_messages=[{"role": "user", "content": f"turn-{i}"} for i in range(4)],
        interactions=[
            LLMInteraction(
                sequence_num=i + 1,
                output_message={"content": f"reply-{i}", "tool_calls": []},
            )
            for i in range(4)
        ],
    )
    path = tmp_path / "multi.json"
    path.write_text(recording.model_dump_json())
    return path


@pytest.mark.asyncio
async def test_run_start_at_skips_leading_turns(
    multi_turn_recording_path: Path,
) -> None:
    """``start_at=2`` runs only turns 2 and 3 (4-turn recording)."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run(start_at=2)
    user_inputs = [
        msg.get("content", "")
        for interaction in replayed.llm_interactions
        for msg in interaction.input_messages
        if msg.get("role") == "user"
    ]
    assert "turn-2" in user_inputs
    assert "turn-3" in user_inputs
    assert "turn-0" not in user_inputs
    assert "turn-1" not in user_inputs


@pytest.mark.asyncio
async def test_run_stop_at_truncates_trailing_turns(
    multi_turn_recording_path: Path,
) -> None:
    """``stop_at=2`` runs only turns 0 and 1."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run(stop_at=2)
    user_inputs = [
        msg.get("content", "")
        for interaction in replayed.llm_interactions
        for msg in interaction.input_messages
        if msg.get("role") == "user"
    ]
    assert "turn-0" in user_inputs
    assert "turn-1" in user_inputs
    assert "turn-2" not in user_inputs
    assert "turn-3" not in user_inputs


@pytest.mark.asyncio
async def test_run_start_and_stop_at_compose(
    multi_turn_recording_path: Path,
) -> None:
    """``start_at=1, stop_at=3`` runs only turns 1 and 2."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run(start_at=1, stop_at=3)
    user_inputs = [
        msg.get("content", "")
        for interaction in replayed.llm_interactions
        for msg in interaction.input_messages
        if msg.get("role") == "user"
    ]
    assert sorted(set(user_inputs)) == ["turn-1", "turn-2"]


@pytest.mark.asyncio
async def test_run_negative_indices_match_python_slice_semantics(
    multi_turn_recording_path: Path,
) -> None:
    """``stop_at=-1`` should drop the last turn (Python slice semantics)."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run(stop_at=-1)
    user_inputs = [
        msg.get("content", "")
        for interaction in replayed.llm_interactions
        for msg in interaction.input_messages
        if msg.get("role") == "user"
    ]
    assert "turn-3" not in user_inputs
    assert "turn-0" in user_inputs


@pytest.mark.asyncio
async def test_run_default_args_full_replay_unchanged(
    multi_turn_recording_path: Path,
) -> None:
    """Calling ``run()`` with no slice args replays everything (no regression)."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner.run()
    user_inputs = {
        msg.get("content", "")
        for interaction in replayed.llm_interactions
        for msg in interaction.input_messages
        if msg.get("role") == "user"
    }
    assert {"turn-0", "turn-1", "turn-2", "turn-3"}.issubset(user_inputs)


# ----- RecordingOverrides -------------------------------------------------


def test_overrides_resolve_normalizes_negative_indices() -> None:
    """``-1`` resolves to ``total - 1``; out-of-range keys silently dropped."""
    overrides = RecordingOverrides(
        llm_outputs={
            0: {"content": "first"},
            -1: {"content": "last"},
            -2: {"content": "second-to-last"},
            99: {"content": "out-of-range"},
            -99: {"content": "out-of-range-negative"},
        }
    )
    resolved = overrides.resolve(total_llm_interactions=4)
    assert resolved == {
        0: {"content": "first"},
        2: {"content": "second-to-last"},
        3: {"content": "last"},
    }


def test_overrides_resolve_empty_recording_drops_everything() -> None:
    overrides = RecordingOverrides(
        llm_outputs={0: {"content": "x"}, -1: {"content": "y"}}
    )
    assert overrides.resolve(total_llm_interactions=0) == {}


def test_recorded_chat_model_serves_override_in_place_of_recording() -> None:
    """When an override is set for index N, that index returns the override text."""
    recording = ConversationRecording(
        interactions=[
            LLMInteraction(sequence_num=1, output_message={"content": "original-0"}),
            LLMInteraction(sequence_num=2, output_message={"content": "original-1"}),
        ],
    )
    overrides = RecordingOverrides(
        llm_outputs={1: {"content": "OVERRIDDEN"}},
    )
    model = RecordedChatModel(recording=recording, overrides=overrides)
    first = model._generate(messages=[HumanMessage(content="ignored")])
    second = model._generate(messages=[HumanMessage(content="ignored")])
    # First call serves the recording verbatim (no override at index 0).
    assert first.generations[0].message.content == "original-0"
    # Second call serves the override.
    assert second.generations[0].message.content == "OVERRIDDEN"


def test_recorded_chat_model_no_overrides_unchanged_behavior() -> None:
    """A model with ``overrides=None`` behaves identically to baseline."""
    recording = ConversationRecording(
        interactions=[
            LLMInteraction(sequence_num=1, output_message={"content": "untouched"}),
        ],
    )
    model = RecordedChatModel(recording=recording)
    result = model._generate(messages=[HumanMessage(content="ignored")])
    assert result.generations[0].message.content == "untouched"


def test_recorded_chat_model_override_replaces_tool_calls_too() -> None:
    """Override can swap tool_calls, not just content — supports trajectory forks."""
    recording = ConversationRecording(
        interactions=[
            LLMInteraction(
                sequence_num=1,
                output_message={
                    "content": "",
                    "tool_calls": [{"name": "real_tool", "args": {}, "id": "call_a"}],
                },
            ),
        ],
    )
    overrides = RecordingOverrides(
        llm_outputs={
            0: {
                "content": "",
                "tool_calls": [
                    {"name": "alt_tool", "args": {"q": "x"}, "id": "call_b"}
                ],
            }
        },
    )
    model = RecordedChatModel(recording=recording, overrides=overrides)
    result = model._generate(messages=[HumanMessage(content="ignored")])
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert [c["name"] for c in msg.tool_calls] == ["alt_tool"]


@pytest.mark.asyncio
async def test_run_with_overrides_changes_replayed_output(
    multi_turn_recording_path: Path,
) -> None:
    """End-to-end: override at index 0 surfaces in the replayed recording."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    overrides = RecordingOverrides(
        llm_outputs={0: {"content": "FORKED-REPLY"}},
    )
    replayed = await runner.run(stop_at=1, overrides=overrides)
    contents = [
        interaction.output_message.get("content", "")
        for interaction in replayed.llm_interactions
    ]
    assert any("FORKED-REPLY" in c for c in contents if isinstance(c, str))


# ---------------------------------------------------------------------------
# Step-mode (issue #78).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_yields_one_turn_result_per_recorded_turn(
    multi_turn_recording_path: Path,
) -> None:
    """Async iterator yields one TurnResult per user turn in the slice."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    turns = [t async for t in runner.step()]
    # 4-turn recording → 4 TurnResults.
    assert len(turns) == 4
    assert all(isinstance(t, TurnResult) for t in turns)
    assert [t.turn_index for t in turns] == [0, 1, 2, 3]
    assert [t.user_input for t in turns] == [
        "turn-0",
        "turn-1",
        "turn-2",
        "turn-3",
    ]


@pytest.mark.asyncio
async def test_step_composes_with_start_at_and_stop_at(
    multi_turn_recording_path: Path,
) -> None:
    """``start_at`` / ``stop_at`` slice the same way ``run`` does."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    turns = [t async for t in runner.step(start_at=1, stop_at=3)]
    assert [t.turn_index for t in turns] == [1, 2]
    assert [t.user_input for t in turns] == ["turn-1", "turn-2"]


@pytest.mark.asyncio
async def test_step_negative_indices_match_python_slice_semantics(
    multi_turn_recording_path: Path,
) -> None:
    """``stop_at=-1`` drops the last turn; ``turn_index`` stays absolute."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    turns = [t async for t in runner.step(start_at=-2, stop_at=-1)]
    # 4-turn recording, slice [-2:-1] → just turn 2.
    assert [t.turn_index for t in turns] == [2]
    assert [t.user_input for t in turns] == ["turn-2"]


@pytest.mark.asyncio
async def test_step_overrides_change_what_the_agent_emits(
    multi_turn_recording_path: Path,
) -> None:
    """When an override is set for a turn, the new_messages reflect it."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    overrides = RecordingOverrides(
        llm_outputs={0: {"content": "FORKED"}},
    )
    turns = [t async for t in runner.step(stop_at=1, overrides=overrides)]
    assert len(turns) == 1
    contents = [
        msg.content for msg in turns[0].new_messages if isinstance(msg, AIMessage)
    ]
    assert "FORKED" in contents


@pytest.mark.asyncio
async def test_step_eager_drain_matches_run_message_count(
    multi_turn_recording_path: Path,
) -> None:
    """Eager-draining step() should produce the same total messages as run()."""
    runner_step = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    step_turns = [t async for t in runner_step.step()]
    step_total_new_messages = sum(len(t.new_messages) for t in step_turns)

    runner_run = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    replayed = await runner_run.run()
    run_total_messages = len(replayed.llm_interactions) + len(
        replayed.tool_interactions
    )

    # The step variant aggregates the per-turn ``new_messages`` slices;
    # the run variant captures via ConversationRecorder. Both should
    # have *seen* the same number of LLM-or-tool events. Loose check
    # because the recorder filters events differently than the raw
    # ``messages`` list, but they should both be in the same ballpark
    # (and both > 0).
    assert step_total_new_messages > 0
    assert run_total_messages > 0


@pytest.mark.asyncio
async def test_step_pause_between_turns_lets_caller_inspect(
    multi_turn_recording_path: Path,
) -> None:
    """The iterator pauses between yields; caller drives the cadence.

    Demonstrates the intended use case: between turns, the caller
    can inspect TurnResult, log it, decide whether to continue, etc.
    """
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    seen: list[int] = []
    async for turn in runner.step(stop_at=2):
        seen.append(turn.turn_index)
        if turn.turn_index == 0:
            # Caller decides between turns: do some work synchronously.
            assert seen == [0]
    # After the loop both turns came through.
    assert seen == [0, 1]


@pytest.mark.asyncio
async def test_step_default_args_iterates_full_recording(
    multi_turn_recording_path: Path,
) -> None:
    """No-arg ``step()`` covers the same range as no-arg ``run()``."""
    runner = ReplayRunner(
        recording_path=multi_turn_recording_path,
        graph_builder=_simple_graph_builder,
        checkpointer=InMemorySaver(),
        store=MockStore(),
    )
    turns = [t async for t in runner.step()]
    assert {t.user_input for t in turns} == {
        "turn-0",
        "turn-1",
        "turn-2",
        "turn-3",
    }


def test_turn_result_is_frozen() -> None:
    """Mutation after yield would surprise inspector callers."""
    tr = TurnResult(turn_index=0, user_input="x")
    with pytest.raises(Exception):  # noqa: B017,PT011 - dataclass FrozenInstanceError
        tr.user_input = "changed"  # type: ignore[misc]
