"""Pairwise LLM judge — compares two outputs and picks a winner.

Why pairwise instead of absolute scoring?

Absolute scores from a single judge are noisy and prone to drift across
runs, prompts, and models. Pairwise comparisons (A vs B for the same
input) are far more stable: the judge only has to answer "which is
better and why," not "score this 0-1." Anthropic's prompt-engineering
guidance and most open evaluation literature converges on pairwise
preference as the lower-variance signal.

Bias guards baked in
--------------------
- **Order randomization.** A and B are randomly swapped per call so the
  judge can't develop a position bias. The shuffle is recorded so we
  un-shuffle when aggregating.
- **Multi-judge agreement.** A pair is only "decided" when N judges
  (default 2) agree. Disagreements count as "no decision" so judge
  bias doesn't leak into the win rate.
- **Length normalization clause.** The default rubric tells the judge
  to ignore response length. Operators are still expected to monitor
  ``ResponseLengthMetric`` separately.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgraph_kit.evals.metrics.model_graded import _extract_json

if TYPE_CHECKING:
    from langgraph_kit.evals.models import TraceData

logger = logging.getLogger(__name__)


_PAIRWISE_SYSTEM = (
    "You are an evaluation judge. Compare two assistant outputs (A and B) "
    "for the same input and decide which response is better according to "
    "the rubric. Reply with a single JSON object and nothing else: "
    '{"winner": "A" | "B" | "tie", "confidence": <float 0.0-1.0>, '
    '"reason": "<one-sentence justification>"}. '
    "Do not wrap the JSON in markdown. Do not include any other keys. "
    "Ignore response length differences when judging quality unless the "
    "rubric specifically calls them out."
)


_DEFAULT_RUBRIC = """\
Compare the two assistant outputs against the user input and decide
which response is more helpful, accurate, and well-structured.

Rules:
- Pick A or B if one is clearly better.
- Pick "tie" only when the responses are genuinely indistinguishable on
  the criteria above.
- Do not favor longer responses.
- Penalize hallucinations or unsupported claims regardless of fluency.

Input:
{{input}}

Output A:
{{output_a}}

Output B:
{{output_b}}
"""


@dataclass(frozen=True)
class PairwiseDecision:
    """One judge's verdict on one A/B pair."""

    judge_name: str
    winner: Literal["A", "B", "tie"]
    confidence: float
    reason: str


@dataclass(frozen=True)
class PairwiseResult:
    """Aggregated verdict across all judges for one A/B pair.

    A pair is *decided* only when every judge agrees on a non-tie winner
    OR every judge agrees on a tie. Mixed results are *undecided*.
    """

    decisions: list[PairwiseDecision]
    decided: bool
    winner: Literal["A", "B", "tie", "undecided"]
    base_was_a: bool

    @property
    def base_won(self) -> bool:
        if not self.decided or self.winner == "tie":
            return False
        if self.base_was_a:
            return self.winner == "A"
        return self.winner == "B"

    @property
    def variant_won(self) -> bool:
        if not self.decided or self.winner == "tie":
            return False
        if self.base_was_a:
            return self.winner == "B"
        return self.winner == "A"


class PairwiseJudge:
    """A single LLM-backed judge for pairwise comparison.

    Use :class:`PairwisePanel` to combine multiple judges with
    agreement-based decision logic.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        rubric_path: str | Path | None = None,
        rubric_text: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self._llm = llm
        if rubric_text is not None:
            self._rubric = rubric_text
        elif rubric_path is not None:
            self._rubric = Path(rubric_path).read_text(encoding="utf-8")
        else:
            self._rubric = _DEFAULT_RUBRIC

    async def judge(
        self,
        input_text: str,
        output_a: str,
        output_b: str,
    ) -> PairwiseDecision:
        prompt = (
            self._rubric.replace("{{input}}", input_text)
            .replace("{{output_a}}", output_a)
            .replace("{{output_b}}", output_b)
        )

        try:
            from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
                HumanMessage,
                SystemMessage,
            )

            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=_PAIRWISE_SYSTEM),
                    HumanMessage(content=prompt),
                ]
            )
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = _extract_json(content)
        except Exception:
            logger.exception("Pairwise judge %r failed to invoke LLM", self.name)
            return PairwiseDecision(
                judge_name=self.name,
                winner="tie",
                confidence=0.0,
                reason="judge_failed",
            )

        winner = str(parsed.get("winner", "tie")).strip().lower()
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason", ""))[:240]

        # Normalize to upper-case A/B/tie label
        normalized: Literal["A", "B", "tie"]
        if winner == "a":
            normalized = "A"
        elif winner == "b":
            normalized = "B"
        else:
            normalized = "tie"

        return PairwiseDecision(
            judge_name=self.name,
            winner=normalized,
            confidence=confidence,
            reason=reason,
        )


class PairwisePanel:
    """Multi-judge panel that produces a decided/undecided verdict."""

    def __init__(self, judges: list[PairwiseJudge], rng: random.Random | None = None) -> None:
        if not judges:
            msg = "PairwisePanel requires at least one judge"
            raise ValueError(msg)
        super().__init__()
        self._judges = judges
        self._rng = rng or random.Random()

    async def compare(
        self,
        input_text: str,
        base_trace: TraceData,
        variant_trace: TraceData,
    ) -> PairwiseResult:
        """Compare base vs variant; randomize A/B placement."""
        base_text = _trace_output_text(base_trace)
        variant_text = _trace_output_text(variant_trace)

        # Randomize A/B placement to avoid any positional bias.
        base_was_a = self._rng.choice([True, False])
        if base_was_a:
            output_a, output_b = base_text, variant_text
        else:
            output_a, output_b = variant_text, base_text

        decisions: list[PairwiseDecision] = []
        for judge in self._judges:
            decision = await judge.judge(input_text, output_a, output_b)
            decisions.append(decision)

        winners = {d.winner for d in decisions}
        decided = len(winners) == 1
        # ``winners`` is a set of ``Literal["A","B","tie"]`` so the
        # element below is one of those three; basedpyright loses that
        # narrowing across ``next(iter(...))``, hence the type-ignore.
        winner: Literal["A", "B", "tie", "undecided"]
        if decided:  # noqa: SIM108 - basedpyright loses Literal narrowing through next()
            winner = next(iter(winners))  # type: ignore[assignment]
        else:
            winner = "undecided"

        return PairwiseResult(
            decisions=decisions,
            decided=decided,
            winner=winner,
            base_was_a=base_was_a,
        )


def _trace_output_text(trace: TraceData) -> str:
    """Best-effort extraction of an assistant output string from TraceData."""
    output = trace.output
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        # Common shapes: {"content": "..."} or {"messages": [{"content": "..."}, ...]}
        if "content" in output and isinstance(output["content"], str):
            return output["content"]
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
        return json.dumps(output, ensure_ascii=False)
    return str(output)
