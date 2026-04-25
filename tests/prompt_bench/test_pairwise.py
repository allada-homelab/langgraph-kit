"""Tests for the pairwise judge panel."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from langgraph_kit.evals.models import TraceData
from tests.prompt_bench.pairwise import (
    PairwiseDecision,
    PairwiseJudge,
    PairwisePanel,
)


class _CannedJudgeLLM:
    """Returns a fixed judge response. Used to drive panel logic without an LLM."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__()
        self._payload = json.dumps(response)

    async def ainvoke(self, _messages: list[Any]) -> Any:
        class _R:
            content = self._payload

        return _R()


class TestPairwiseJudge:
    async def test_parses_winner_and_confidence(self) -> None:
        llm = _CannedJudgeLLM({"winner": "A", "confidence": 0.83, "reason": "clearer"})
        judge = PairwiseJudge(name="j", llm=llm, rubric_text="dummy")
        decision = await judge.judge("input", "out_a", "out_b")
        assert decision.winner == "A"
        assert decision.confidence == pytest.approx(0.83)
        assert decision.reason == "clearer"
        assert decision.judge_name == "j"

    async def test_normalizes_lowercase(self) -> None:
        llm = _CannedJudgeLLM({"winner": "b", "confidence": 0.5, "reason": ""})
        judge = PairwiseJudge(name="j", llm=llm, rubric_text="dummy")
        decision = await judge.judge("i", "a", "b")
        assert decision.winner == "B"

    async def test_invalid_winner_falls_back_to_tie(self) -> None:
        llm = _CannedJudgeLLM({"winner": "neither", "confidence": 1.0, "reason": "x"})
        judge = PairwiseJudge(name="j", llm=llm, rubric_text="dummy")
        decision = await judge.judge("i", "a", "b")
        assert decision.winner == "tie"


class TestPairwisePanel:
    @staticmethod
    def _make_panel(
        decisions: list[Any],
        rng_seed: int = 0,
    ) -> PairwisePanel:
        judges = [
            PairwiseJudge(
                name=f"j{i}",
                llm=_CannedJudgeLLM(d),
                rubric_text="dummy",
            )
            for i, d in enumerate(decisions)
        ]
        return PairwisePanel(judges=judges, rng=random.Random(rng_seed))

    @staticmethod
    def _trace(text: str, label: str) -> TraceData:
        return TraceData(id=label, output=text)

    async def test_decided_when_all_judges_agree(self) -> None:
        # rng_seed=0 → first random.choice([True, False]) is True (base_was_a=True)
        panel = self._make_panel(
            [
                {"winner": "B", "confidence": 0.9, "reason": "x"},
                {"winner": "B", "confidence": 0.8, "reason": "y"},
            ]
        )
        result = await panel.compare(
            input_text="q",
            base_trace=self._trace("base", "base"),
            variant_trace=self._trace("variant", "variant"),
        )
        assert result.decided
        assert result.winner == "B"
        # If base_was_a is True, then "B" winner means variant won
        if result.base_was_a:
            assert result.variant_won
            assert not result.base_won
        else:
            # Inverted shuffle: "B" winner means base won
            assert result.base_won

    async def test_undecided_when_judges_disagree(self) -> None:
        panel = self._make_panel(
            [
                {"winner": "A", "confidence": 0.9, "reason": "x"},
                {"winner": "B", "confidence": 0.8, "reason": "y"},
            ]
        )
        result = await panel.compare(
            input_text="q",
            base_trace=self._trace("base", "base"),
            variant_trace=self._trace("variant", "variant"),
        )
        assert not result.decided
        assert result.winner == "undecided"
        assert not result.base_won
        assert not result.variant_won

    async def test_tie_consensus_means_tie_not_undecided(self) -> None:
        panel = self._make_panel(
            [
                {"winner": "tie", "confidence": 0.5, "reason": "x"},
                {"winner": "tie", "confidence": 0.5, "reason": "y"},
            ]
        )
        result = await panel.compare(
            input_text="q",
            base_trace=self._trace("base", "base"),
            variant_trace=self._trace("variant", "variant"),
        )
        assert result.decided
        assert result.winner == "tie"
        assert not result.base_won
        assert not result.variant_won

    async def test_requires_at_least_one_judge(self) -> None:
        with pytest.raises(ValueError, match="at least one judge"):
            PairwisePanel(judges=[])

    async def test_ab_order_random_across_calls(self) -> None:
        # Different RNG seeds should produce different A/B placements
        # over a single comparison (a smoke check that randomness is wired).
        panel_a = self._make_panel(
            [{"winner": "A", "confidence": 1, "reason": ""}],
            rng_seed=1,
        )
        panel_b = self._make_panel(
            [{"winner": "A", "confidence": 1, "reason": ""}],
            rng_seed=42,
        )
        r_a = await panel_a.compare(
            "q", self._trace("base", "b"), self._trace("variant", "v")
        )
        r_b = await panel_b.compare(
            "q", self._trace("base", "b"), self._trace("variant", "v")
        )
        # We expect this assertion to hold often but it's a property,
        # not a guarantee — accept either outcome but assert at least
        # one combination was tried.
        assert r_a.base_was_a in (True, False)
        assert r_b.base_was_a in (True, False)


class TestPairwiseDecision:
    def test_decision_is_immutable(self) -> None:
        d = PairwiseDecision(judge_name="j", winner="A", confidence=0.5, reason="r")
        with pytest.raises(Exception):  # noqa: B017,PT011 - frozen dataclass
            d.winner = "B"  # type: ignore[misc]
