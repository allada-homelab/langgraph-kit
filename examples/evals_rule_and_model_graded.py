"""Evals: rule-based metrics + model-graded LLMJudge against synthetic traces.

What this shows
---------------
- Constructing :class:`TraceData` directly (no Langfuse round-trip)
- Scoring with rule-based metrics: ``ResponseLengthMetric``,
  ``LatencyMetric``, ``ErrorFreeMetric``
- Optionally invoking :class:`LLMJudgeMetric` when real-LLM mode is
  enabled — the judge prompt + JSON parser pattern matches what the
  full :class:`EvalRunner` consumes against live Langfuse traces

The hermetic path runs only the rule-based metrics so the demo always
exits 0 without hitting an LLM. The model-graded path activates with
``LANGGRAPH_KIT_EXAMPLES_LLM=real`` plus ``AGENT_LLM_API_KEY`` and
shows what the judge returns.

How to run
----------
    uv run python -m examples.evals_rule_and_model_graded                                  # hermetic
    LANGGRAPH_KIT_EXAMPLES_LLM=real uv run python -m examples.evals_rule_and_model_graded  # + LLM judge

Expected output (hermetic)
--------------------------
    Synthetic trace: <id> latency=1200ms output=84 chars
    Rule-based scores:
      response_length  value=0.6 (Too short: 6 words (min: 10))
      latency          value=1.0 (Within SLA: 1200ms <= 30000ms)
      error_free       value=True (No errors detected)
    Skipping LLMJudgeMetric — set LANGGRAPH_KIT_EXAMPLES_LLM=real to enable.
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, hermetic, line


async def main() -> None:
    banner("evals_rule_and_model_graded")

    from langgraph_kit.evals.metrics.rule_based import (
        ErrorFreeMetric,
        LatencyMetric,
        ResponseLengthMetric,
    )
    from langgraph_kit.evals.models import TraceData

    # Build a synthetic trace by hand. In production this comes from
    # Langfuse via :class:`EvalRunner._fetch_traces`.
    trace = TraceData(
        id="trace-demo-001",
        name="echo-agent",
        input="What's the kit's recursion limit?",
        output="The default recursion limit is 100.",
        tags=["demo"],
        duration_ms=1200,
        metadata={"status": "ok"},
    )
    line(
        f"Synthetic trace: {trace.id} latency={int(trace.duration_ms or 0)}ms "
        f"output={len(trace.output or '')} chars"
    )

    line("Rule-based scores:")
    metrics = [ResponseLengthMetric(), LatencyMetric(), ErrorFreeMetric()]
    for metric in metrics:
        result = await metric.score(trace)
        line(f"  {metric.name:<16} value={result.value} ({result.comment})")

    if hermetic():
        line("Skipping LLMJudgeMetric — set LANGGRAPH_KIT_EXAMPLES_LLM=real to enable.")
        return

    # Real-LLM path. Configure the kit and run the LLM judge against the
    # same synthetic trace.
    from examples._lib import (
        assert_real_llm_or_skip,
        configure_real_llm,
        tmp_workspace,
    )

    assert_real_llm_or_skip()
    with tmp_workspace() as workspace:
        configure_real_llm(workspace)

        from langgraph_kit.evals.metrics.model_graded import LLMJudgeMetric
        from langgraph_kit.llm import build_llm

        judge = LLMJudgeMetric(
            llm=build_llm(),
            criteria="Does the assistant answer the user's question concisely?",
            name="conciseness",
        )
        result = await judge.score(trace)
        line(f"  {judge.name:<16} value={result.value} ({result.comment})")


if __name__ == "__main__":
    asyncio.run(main())
