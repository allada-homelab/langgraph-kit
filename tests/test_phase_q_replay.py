"""Regression tests for Phase Q replay polish."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from pathlib import Path

from langgraph_kit.replay.models import ConversationRecording, LLMInteraction
from langgraph_kit.replay.player import (
    RecordedChatModel,
    ReplayMismatchError,
)
from langgraph_kit.replay.runner import ReplayRunner

if TYPE_CHECKING:
    from pathlib import Path as _Path  # noqa: F401


def _one_interaction_recording() -> ConversationRecording:
    return ConversationRecording(
        agent_id="a",
        thread_id="t",
        interactions=[
            LLMInteraction(
                sequence_num=0,
                kind="llm",
                input_messages=[{"role": "user", "content": "hi"}],
                output_message={"content": "hello"},
                model_name="m",
            )
        ],
        user_messages=[],
    )


def test_recorded_model_strict_mode_raises_on_second_call() -> None:
    """When ``fuzzy_match=False``, exhausting the sequence raises
    ReplayMismatchError on the next call — no silent re-serve."""
    rec = _one_interaction_recording()
    model = RecordedChatModel(recording=rec, fuzzy_match=False)
    # First call consumes the one recorded interaction.
    model._generate([HumanMessage(content="hi")])
    # Second call should raise.
    with pytest.raises(ReplayMismatchError):
        model._generate([HumanMessage(content="hi")])


def test_recorded_model_fuzzy_default_still_reserves() -> None:
    rec = _one_interaction_recording()
    model = RecordedChatModel(recording=rec)  # fuzzy_match default True
    model._generate([HumanMessage(content="hi")])
    # Second call: the fuzzy fallback re-serves the same interaction.
    result = model._generate([HumanMessage(content="hi")])
    assert any("hello" in gen.text for gen in result.generations)


def test_runner_uses_llm_kwarg_name(tmp_path: Path) -> None:
    """ReplayRunner's ``llm_kwarg`` should control which name the mock
    LLM is passed through to graph_builder."""
    rec = _one_interaction_recording()
    fixture = tmp_path / "rec.json"
    fixture.write_text(json.dumps(rec.model_dump(mode="json")), encoding="utf-8")

    captured: dict[str, Any] = {}

    def _builder(_ckpt: Any, _store: Any, **kwargs: Any) -> Any:
        # Record which kwarg the runner used.
        captured.update(kwargs)

        class _Graph:
            async def ainvoke(self, inp: Any, config: Any) -> dict[str, Any]:
                return {"messages": []}

        return _Graph()

    runner = ReplayRunner(
        recording_path=fixture,
        graph_builder=_builder,
        llm_kwarg="model",  # non-default
    )

    import asyncio

    asyncio.run(runner.run())
    assert "model" in captured, f"Expected ``model=`` kwarg; got {captured!r}"
    assert "llm" not in captured


def test_runner_defaults_to_llm_kwarg(tmp_path: Path) -> None:
    rec = _one_interaction_recording()
    fixture = tmp_path / "rec.json"
    fixture.write_text(json.dumps(rec.model_dump(mode="json")), encoding="utf-8")

    captured: dict[str, Any] = {}

    def _builder(_ckpt: Any, _store: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)

        class _Graph:
            async def ainvoke(self, inp: Any, config: Any) -> dict[str, Any]:
                return {"messages": []}

        return _Graph()

    runner = ReplayRunner(recording_path=fixture, graph_builder=_builder)

    import asyncio

    asyncio.run(runner.run())
    assert "llm" in captured
