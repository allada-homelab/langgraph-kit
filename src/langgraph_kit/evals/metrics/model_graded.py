"""Model-graded evaluation metrics — uses an LLM as judge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langgraph_kit.evals.models import EvalMetric, EvalResult, TraceData

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_JUDGE_SYSTEM = (
    "You are an evaluation judge. Score the interaction against the rubric "
    "and reply with a single JSON object and nothing else: "
    '{"score": <float 0.0-1.0>, "reason": "<one-sentence justification>"}. '
    "Do not wrap the JSON in markdown. Do not include any other keys. "
    "The score must be a number between 0.0 and 1.0 inclusive."
)


class LLMJudgeMetric(EvalMetric):
    """Evaluate traces using an LLM with a rubric prompt.

    Loads a rubric from a ``.md`` file that uses ``{{input}}`` and
    ``{{output}}`` placeholders.  Calls the provided LLM and parses
    a JSON response with ``score`` (0-1) and ``reason`` fields.
    """

    data_type = "NUMERIC"

    def __init__(
        self,
        name: str,
        rubric_path: str | Path | None = None,
        rubric_text: str | None = None,
        llm: Any = None,
    ) -> None:
        super().__init__()
        self.name = name
        self._llm = llm
        if rubric_text:
            self._rubric = rubric_text
        elif rubric_path:
            self._rubric = Path(rubric_path).read_text(encoding="utf-8")
        else:
            # Try default prompts directory
            default = _PROMPTS_DIR / f"{name}.md"
            if default.is_file():
                self._rubric = default.read_text(encoding="utf-8")
            else:
                msg = f"No rubric provided for metric '{name}' and no default found at {default}"
                raise FileNotFoundError(msg)

    async def score(self, trace: TraceData) -> EvalResult:
        if self._llm is None:
            return EvalResult(value=0.5, comment="No LLM configured for judging")

        prompt = self._rubric.replace(
            "{{input}}", str(trace.input or "(no input)")
        ).replace("{{output}}", str(trace.output or "(no output)"))

        try:
            from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
                HumanMessage,
                SystemMessage,
            )

            response = await self._llm.ainvoke(
                [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
            )
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            parsed = _extract_json(content)
            if "score" not in parsed:
                logger.warning(
                    "LLM judge for '%s' returned no 'score' field; parsed=%r",
                    self.name,
                    parsed,
                )
                return EvalResult(
                    value=0.0, comment="Judge returned no score field"
                )
            try:
                score_val = float(parsed["score"])
            except (TypeError, ValueError):
                score_val = 0.5
            score_val = max(0.0, min(1.0, score_val))
            reason = str(parsed.get("reason", ""))
            return EvalResult(value=round(score_val, 3), comment=reason)
        except Exception:
            logger.exception("LLM judge failed for metric '%s'", self.name)
            return EvalResult(value=0.0, comment="LLM judge evaluation failed")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output that may contain markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass
    return {"score": 0.5, "reason": "Could not parse LLM judge output"}
