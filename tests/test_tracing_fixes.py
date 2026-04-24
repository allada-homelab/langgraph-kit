"""Regression tests for Phase H tracing fixes."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.tracing.mermaid import _safe_name, trace_to_mermaid
from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan
from langgraph_kit.core.tracing.storage import TraceStore

from .conftest import MockStore


def _span(
    *, span_id: str, name: str = "n", kind: str = "chain", children: list[Any] | None = None
) -> TraceSpan:
    return TraceSpan(
        span_id=span_id,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        started_at="2026-04-24T00:00:00Z",
        ended_at="2026-04-24T00:00:01Z",
        duration_ms=1000.0,
        children=children or [],
    )


def test_flowchart_renders_children_with_parent_child_edges() -> None:
    leaf = _span(span_id="s3", name="leaf", kind="tool")
    mid = _span(span_id="s2", name="mid", kind="llm", children=[leaf])
    root = _span(span_id="s1", name="root", kind="chain", children=[mid])

    trace = TraceRecord(
        trace_id="t",
        started_at="2026-04-24T00:00:00Z",
        ended_at="2026-04-24T00:00:01Z",
        duration_ms=1000.0,
        spans=[root],
    )
    out = trace_to_mermaid(trace, style="flowchart")
    # Three nodes (one per span) and at least two parent->child edges.
    assert out.count('["') == 3, f"Expected 3 nodes, got:\n{out}"
    edge_lines = [line for line in out.splitlines() if "-->" in line]
    assert len(edge_lines) >= 2, f"Expected ≥2 edges, got:\n{out}"


def test_safe_name_escapes_mermaid_reserved_characters() -> None:
    # Brackets, pipes, backticks, and arrow literal are all sanitized.
    cleaned = _safe_name("call [tool] `do` --> result | next")
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "|" not in cleaned
    assert "`" not in cleaned
    assert "-->" not in cleaned


def test_safe_name_truncates_to_50_chars() -> None:
    assert len(_safe_name("x" * 200)) == 50


@pytest.mark.asyncio
async def test_get_trace_uses_direct_aget(monkeypatch: Any) -> None:
    store = MockStore()
    tstore = TraceStore(store)
    trace = TraceRecord(
        trace_id="trace-1",
        started_at="2026-04-24T00:00:00Z",
        ended_at="2026-04-24T00:00:01Z",
        duration_ms=1000.0,
    )
    await tstore.save_trace("thread-a", trace)

    # Patch asearch to raise — if get_trace falls back to it, this fails.
    async def _no_search(*_a: Any, **_kw: Any) -> list[Any]:
        raise AssertionError("get_trace must not fall back to asearch")

    monkeypatch.setattr(store, "asearch", _no_search)
    out = await tstore.get_trace("thread-a", "trace-1")
    assert out is not None
    assert out.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_list_traces_reads_materialized_summaries(
    monkeypatch: Any,
) -> None:
    """list_traces should consume the .summary companion keys and not
    deserialize full span trees for every trace."""
    store = MockStore()
    tstore = TraceStore(store)

    big_span = _span(span_id="s", name="n")
    traces = [
        TraceRecord(
            trace_id=f"t{i}",
            started_at=f"2026-04-24T00:00:{i:02d}Z",
            ended_at=f"2026-04-24T00:00:{i + 1:02d}Z",
            duration_ms=1000.0,
            spans=[big_span],
        )
        for i in range(3)
    ]
    for t in traces:
        await tstore.save_trace("tid", t)

    # Count how many times TraceRecord.model_validate is called during
    # list_traces — the summaries path should not call it at all.
    from langgraph_kit.core.tracing import storage as storage_mod

    calls = {"n": 0}
    real_validate = storage_mod.TraceRecord.model_validate

    @classmethod  # type: ignore[misc]
    def counting_validate(cls: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        storage_mod.TraceRecord, "model_validate", counting_validate
    )

    summaries = await tstore.list_traces("tid")
    assert len(summaries) == 3
    assert calls["n"] == 0, (
        "list_traces should read the .summary companion keys and not "
        "materialise full TraceRecord blobs."
    )
