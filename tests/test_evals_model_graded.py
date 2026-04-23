"""Coverage fill — ``LLMJudgeMetric`` with a mock LLM.

The metric needs a real LLM at runtime, but every other code path —
rubric loading, prompt placeholder substitution, JSON extraction with
markdown-fence tolerance, score clamping, error-to-0.0 conversion — is
pure logic. These tests drive each branch with a FakeLLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from langgraph_kit.evals.metrics.model_graded import (
    LLMJudgeMetric,
    _extract_json,
)
from langgraph_kit.evals.models import TraceData


class _FakeLLM:
    """Minimal LLM stand-in returning a pre-set JSON response."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def ainvoke(self, messages: list[Any]) -> Any:
        _ = messages

        class _Response:
            content = self._response

        return _Response()


class _RaisingLLM:
    async def ainvoke(self, messages: list[Any]) -> Any:
        _ = messages
        msg = "judge LLM unreachable"
        raise RuntimeError(msg)


def _trace() -> TraceData:
    return TraceData(id="t1", input={"q": "hi"}, output="hello")


# ---------------------------------------------------------------------------
# Construction / rubric loading
# ---------------------------------------------------------------------------


def test_metric_uses_rubric_text_argument_verbatim() -> None:
    metric = LLMJudgeMetric(
        name="custom",
        rubric_text="RUBRIC BODY with {{input}} and {{output}} placeholders",
        llm=None,
    )
    # Private attribute access is intentional — exercising constructor path.
    assert "RUBRIC BODY" in metric._rubric  # pyright: ignore[reportPrivateUsage]


def test_metric_loads_rubric_from_explicit_path(tmp_path: Path) -> None:
    rubric = tmp_path / "custom.md"
    rubric.write_text("hello {{input}} world {{output}}", encoding="utf-8")
    metric = LLMJudgeMetric(name="custom", rubric_path=rubric, llm=None)
    assert "hello" in metric._rubric  # pyright: ignore[reportPrivateUsage]


def test_metric_finds_default_rubric_in_prompts_dir() -> None:
    # ``helpfulness.md`` ships with the kit under the evals prompts dir.
    metric = LLMJudgeMetric(name="helpfulness", llm=None)
    assert metric._rubric  # pyright: ignore[reportPrivateUsage]


def test_metric_raises_when_no_rubric_and_no_default() -> None:
    with pytest.raises(FileNotFoundError, match="No rubric"):
        LLMJudgeMetric(name="totally-made-up-metric-3f7a", llm=None)


# ---------------------------------------------------------------------------
# score() — each branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_no_llm_returns_neutral() -> None:
    metric = LLMJudgeMetric(name="m", rubric_text="r", llm=None)
    result = await metric.score(_trace())
    assert result.value == 0.5
    assert "No LLM" in (result.comment or "")


@pytest.mark.asyncio
async def test_score_parses_plain_json_response() -> None:
    metric = LLMJudgeMetric(
        name="m",
        rubric_text="r",
        llm=_FakeLLM('{"score": 0.85, "reason": "Looks solid"}'),
    )
    result = await metric.score(_trace())
    assert result.value == 0.85
    assert result.comment == "Looks solid"


@pytest.mark.asyncio
async def test_score_clamps_out_of_range_values() -> None:
    # Score >1 clamps to 1; <0 clamps to 0.
    over = LLMJudgeMetric(name="over", rubric_text="r", llm=_FakeLLM('{"score": 1.8}'))
    result = await over.score(_trace())
    assert result.value == 1.0

    under = LLMJudgeMetric(
        name="under", rubric_text="r", llm=_FakeLLM('{"score": -0.3}')
    )
    result = await under.score(_trace())
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_score_handles_non_numeric_score_field() -> None:
    metric = LLMJudgeMetric(
        name="m",
        rubric_text="r",
        llm=_FakeLLM('{"score": "bogus", "reason": "noise"}'),
    )
    result = await metric.score(_trace())
    # Falls back to 0.5 neutral on ValueError/TypeError during float cast.
    assert result.value == 0.5


@pytest.mark.asyncio
async def test_score_missing_score_field_returns_zero() -> None:
    metric = LLMJudgeMetric(
        name="m", rubric_text="r", llm=_FakeLLM('{"reason": "forgot to score"}')
    )
    result = await metric.score(_trace())
    assert result.value == 0.0
    assert "no score" in (result.comment or "").lower()


@pytest.mark.asyncio
async def test_score_llm_exception_returns_zero() -> None:
    metric = LLMJudgeMetric(name="m", rubric_text="r", llm=_RaisingLLM())
    result = await metric.score(_trace())
    assert result.value == 0.0
    assert "failed" in (result.comment or "").lower()


@pytest.mark.asyncio
async def test_score_substitutes_input_output_placeholders() -> None:
    class _CapturingLLM:
        def __init__(self) -> None:
            self.captured_prompt: str = ""

        async def ainvoke(self, messages: list[Any]) -> Any:
            # The rubric-filled prompt is the second message (HumanMessage).
            self.captured_prompt = str(messages[-1].content)

            class _R:
                content = '{"score": 1.0}'

            return _R()

    capturer = _CapturingLLM()
    metric = LLMJudgeMetric(
        name="m",
        rubric_text="input={{input}} output={{output}}",
        llm=capturer,
    )
    trace = TraceData(id="t", input="HELLO", output="WORLD")
    await metric.score(trace)
    assert "input=HELLO" in capturer.captured_prompt
    assert "output=WORLD" in capturer.captured_prompt


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain_object() -> None:
    assert _extract_json('{"score": 0.9}') == {"score": 0.9}


def test_extract_json_tolerates_markdown_fences() -> None:
    fenced = '```json\n{"score": 0.7, "reason": "ok"}\n```'
    assert _extract_json(fenced) == {"score": 0.7, "reason": "ok"}


def test_extract_json_finds_embedded_object_in_prose() -> None:
    noisy = 'Here is my answer: {"score": 0.5} — hope that helps.'
    assert _extract_json(noisy) == {"score": 0.5}


def test_extract_json_falls_back_to_neutral_on_garbage() -> None:
    result = _extract_json("completely not json at all")
    assert result == {"score": 0.5, "reason": "Could not parse LLM judge output"}
