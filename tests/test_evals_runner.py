"""Coverage fill — ``EvalRunner.run`` orchestration logic with a fake Langfuse.

The runner fetches traces from Langfuse, scores them against each
registered metric, posts scores back, and aggregates into an
``EvalReport``. These tests drive the orchestration with a fake
Langfuse client so every branch is exercised (numeric aggregation,
boolean pass_rate, metric failure isolation, dry-run mode, trace-fetch
errors).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langgraph_kit.evals.models import EvalMetric, EvalResult, TraceData
from langgraph_kit.evals.runner import EvalRunner


class _NumericMetric(EvalMetric):
    name = "numeric"
    data_type = "NUMERIC"

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self._values = iter(values)

    async def score(self, trace: TraceData) -> EvalResult:
        _ = trace
        return EvalResult(value=next(self._values), comment="ok")


class _BoolMetric(EvalMetric):
    name = "boolean"
    data_type = "BOOLEAN"

    def __init__(self, values: list[bool]) -> None:
        super().__init__()
        self._values = iter(values)

    async def score(self, trace: TraceData) -> EvalResult:
        _ = trace
        return EvalResult(value=next(self._values))


class _AlwaysFailingMetric(EvalMetric):
    name = "broken"
    data_type = "NUMERIC"

    async def score(self, trace: TraceData) -> EvalResult:
        _ = trace
        msg = "metric internal error"
        raise RuntimeError(msg)


def _fake_trace(tid: str) -> Any:
    """Shape that resembles a Langfuse Trace response."""
    t = MagicMock()
    t.id = tid
    t.name = f"trace-{tid}"
    t.input = "in"
    t.output = "out"
    t.tags = []
    t.latency = 200.0
    t.metadata = {"tool_calls": 1}
    return t


def _fake_langfuse(traces: list[Any], *, fetch_raises: bool = False) -> Any:
    lf = MagicMock()
    if fetch_raises:
        lf.api.trace.list.side_effect = RuntimeError("langfuse unreachable")
    else:
        result = MagicMock()
        result.data = traces
        lf.api.trace.list.return_value = result
    lf.create_score = MagicMock()
    lf.flush = MagicMock()
    return lf


@pytest.mark.asyncio
async def test_run_aggregates_numeric_scores_and_computes_pass_rate() -> None:
    lf = _fake_langfuse([_fake_trace("t1"), _fake_trace("t2"), _fake_trace("t3")])
    metric = _NumericMetric(values=[0.9, 0.5, 0.8])

    runner = EvalRunner(langfuse=lf, metrics=[metric], pass_threshold=0.7)
    report = await runner.run(dry_run=True)

    assert report.total_traces == 3
    summary = report.metrics["numeric"]
    assert summary.count == 3
    assert summary.mean == round((0.9 + 0.5 + 0.8) / 3, 3)
    # 2/3 >= 0.7 pass threshold.
    assert summary.pass_rate == round(2 / 3, 3)


@pytest.mark.asyncio
async def test_run_boolean_metric_pass_rate() -> None:
    lf = _fake_langfuse([_fake_trace("a"), _fake_trace("b"), _fake_trace("c")])
    metric = _BoolMetric(values=[True, False, True])
    report = await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=True)
    summary = report.metrics["boolean"]
    assert summary.pass_rate == round(2 / 3, 3)


@pytest.mark.asyncio
async def test_run_isolates_metric_failures_to_that_metric() -> None:
    """A failing metric on one trace doesn't propagate — other traces / metrics
    keep running. The broken metric's summary count stays at 0; the other
    metric still scores every trace.
    """
    lf = _fake_langfuse([_fake_trace("t1"), _fake_trace("t2")])
    broken = _AlwaysFailingMetric()
    working = _BoolMetric(values=[True, True])

    report = await EvalRunner(
        langfuse=lf, metrics=[broken, working]
    ).run(dry_run=True)

    assert report.metrics["broken"].count == 0
    assert report.metrics["boolean"].count == 2
    assert report.metrics["boolean"].pass_rate == 1.0


@pytest.mark.asyncio
async def test_run_posts_scores_when_not_dry_run() -> None:
    lf = _fake_langfuse([_fake_trace("t1")])
    metric = _NumericMetric(values=[0.95])

    await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=False)
    lf.create_score.assert_called_once()
    lf.flush.assert_called_once()


@pytest.mark.asyncio
async def test_run_skips_langfuse_posts_in_dry_run() -> None:
    lf = _fake_langfuse([_fake_trace("t1")])
    metric = _NumericMetric(values=[0.95])

    await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=True)
    lf.create_score.assert_not_called()
    lf.flush.assert_not_called()


@pytest.mark.asyncio
async def test_run_swallows_create_score_errors() -> None:
    lf = _fake_langfuse([_fake_trace("t1")])
    lf.create_score.side_effect = RuntimeError("post failed")
    metric = _NumericMetric(values=[0.5])

    # Must not raise — the runner catches and logs post failures.
    report = await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=False)
    assert report.total_traces == 1


@pytest.mark.asyncio
async def test_run_returns_empty_report_when_fetch_fails() -> None:
    lf = _fake_langfuse([], fetch_raises=True)
    metric = _NumericMetric(values=[])

    report = await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=True)
    assert report.total_traces == 0
    # Metric summary still created with zero data.
    assert report.metrics["numeric"].count == 0


@pytest.mark.asyncio
async def test_run_handles_langfuse_response_without_data_attr() -> None:
    """If Langfuse returns a bare list instead of ``.data`` shape, it still works."""
    lf = MagicMock()
    # Direct list return — no ``.data`` attr, just the list itself.
    lf.api.trace.list.return_value = [_fake_trace("bare")]
    metric = _BoolMetric(values=[True])
    report = await EvalRunner(langfuse=lf, metrics=[metric]).run(dry_run=True)
    assert report.total_traces == 1


@pytest.mark.asyncio
async def test_run_forwards_tag_filter_to_langfuse_api() -> None:
    lf = _fake_langfuse([])
    await EvalRunner(langfuse=lf, metrics=[]).run(dry_run=True, tags=["agents"])
    kwargs = lf.api.trace.list.call_args.kwargs
    assert kwargs["tags"] == ["agents"]


@pytest.mark.asyncio
async def test_run_sets_duration_seconds_on_report() -> None:
    lf = _fake_langfuse([])
    report = await EvalRunner(langfuse=lf, metrics=[]).run(dry_run=True)
    # duration_seconds is rounded to 2dp; must be a non-negative float.
    assert isinstance(report.duration_seconds, float)
    assert report.duration_seconds >= 0
