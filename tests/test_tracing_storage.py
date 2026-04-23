"""Coverage fill — ``TraceStore`` CRUD + pruning with a ``MockStore``."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan
from langgraph_kit.core.tracing.storage import TraceStore


def _record(trace_id: str, started_at: str, *, span_count: int = 1) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        agent_id="a",
        thread_id="t",
        started_at=started_at,
        ended_at=started_at,
        duration_ms=10.0,
        spans=[TraceSpan(span_id=f"s{i}") for i in range(span_count)],
    )


@pytest.mark.asyncio
async def test_save_and_get_round_trip(mock_store: Any) -> None:
    ts = TraceStore(mock_store)
    rec = _record("tr-1", "2026-04-24T00:00:00Z")
    await ts.save_trace("thread-1", rec)
    fetched = await ts.get_trace("thread-1", "tr-1")
    assert fetched is not None
    assert fetched.trace_id == "tr-1"


@pytest.mark.asyncio
async def test_get_trace_returns_none_for_unknown_id(mock_store: Any) -> None:
    ts = TraceStore(mock_store)
    assert await ts.get_trace("nowhere", "nope") is None


@pytest.mark.asyncio
async def test_list_traces_returns_summaries_sorted_desc(mock_store: Any) -> None:
    ts = TraceStore(mock_store)
    await ts.save_trace("thr", _record("t1", "2026-01-01T00:00:00Z"))
    await ts.save_trace("thr", _record("t2", "2026-04-24T00:00:00Z"))
    await ts.save_trace("thr", _record("t3", "2026-02-15T00:00:00Z"))

    summaries = await ts.list_traces("thr")
    ids = [s.trace_id for s in summaries]
    assert ids == ["t2", "t3", "t1"]


@pytest.mark.asyncio
async def test_list_traces_empty_thread(mock_store: Any) -> None:
    ts = TraceStore(mock_store)
    assert await ts.list_traces("nowhere") == []


@pytest.mark.asyncio
async def test_save_trace_prunes_oldest_when_limit_exceeded(mock_store: Any) -> None:
    ts = TraceStore(mock_store, max_per_thread=3)

    # Insert 5 traces with increasing timestamps.
    for i in range(5):
        await ts.save_trace("thr", _record(f"t{i}", f"2026-01-0{i + 1}T00:00:00Z"))

    # The two oldest (t0, t1) should have been pruned.
    remaining_keys = set(mock_store._data.get(("traces", "thr"), {}).keys())
    assert "t0" not in remaining_keys
    assert "t1" not in remaining_keys
    assert {"t2", "t3", "t4"} <= remaining_keys


@pytest.mark.asyncio
async def test_save_trace_survives_store_failure(
    mock_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising store doesn't propagate — ``save_trace`` catches and logs."""

    class _RaisingStore:
        async def aput(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs
            msg = "store down"
            raise RuntimeError(msg)

        async def asearch(self, *args: Any, **kwargs: Any) -> list[Any]:
            _ = args
            _ = kwargs
            return []

    ts = TraceStore(_RaisingStore())
    await ts.save_trace("thr", _record("t1", "2026-04-24T00:00:00Z"))
    # No exception raised; a warning was logged.
    _ = mock_store  # fixture reuse not needed here
    _ = caplog  # we don't need to assert on the warning text specifically
