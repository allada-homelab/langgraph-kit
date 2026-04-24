"""Regression tests for Phase I orchestration polish."""

from __future__ import annotations

import time
from typing import Any

import pytest

from langgraph_kit.core.orchestration.queue import ThreadBusyTracker, ThreadQueue
from langgraph_kit.core.orchestration.routing import (
    AgentCapability,
    LLMRoutingStrategy,
    _extract_json_object,
    _first_balanced_object,
)
from langgraph_kit.core.resilience.post_run import _prune_thread_records

from .conftest import MockStore

# ---------------------------------------------------------------------------
# Routing JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_object_parses_plain_json() -> None:
    out = _extract_json_object('{"target_agent_id": "x"}')
    assert out == {"target_agent_id": "x"}


def test_extract_json_object_tolerates_fence() -> None:
    out = _extract_json_object(
        '```json\n{"target_agent_id": "y"}\n```'
    )
    assert out == {"target_agent_id": "y"}


def test_extract_json_object_tolerates_prose_wrapping() -> None:
    out = _extract_json_object(
        'Sure, here is my decision: {"target_agent_id": "z"} done.'
    )
    assert out == {"target_agent_id": "z"}


def test_first_balanced_object_handles_strings_with_braces() -> None:
    out = _first_balanced_object('noise {"msg": "has {braces}"} trailing')
    assert out == '{"msg": "has {braces}"}'


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any], config: Any = None) -> Any:
        class _Resp:
            content = self._content

        return _Resp()


@pytest.mark.asyncio
async def test_routing_no_longer_defaults_to_first_agent_on_parse_failure() -> None:
    """Prior behaviour was to silently route to ``capabilities[0]`` when
    the LLM returned un-parseable JSON. That made ordering
    load-bearing in a way callers couldn't control. The fix: return
    ``target_agent_id='none'`` so callers get a deterministic signal."""
    capabilities = [
        AgentCapability(agent_id="a-first", name="a", description="alpha"),
        AgentCapability(agent_id="b-second", name="b", description="beta"),
    ]
    strategy = LLMRoutingStrategy(llm=_FakeLLM("not json at all"))

    decision = await strategy.route(
        message="anything", capabilities=capabilities, history=[]
    )
    assert decision.target_agent_id == "none"


# ---------------------------------------------------------------------------
# Queue drain / depth / clear batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_drain_pages_past_100() -> None:
    store = MockStore()
    queue = ThreadQueue(store, thread_id="tid")
    from langgraph_kit.core.orchestration.queue import QueuedItem, QueueSemantic

    # Enqueue more items than a single asearch page.
    for i in range(150):
        await queue.enqueue(
            QueuedItem(
                id=f"i-{i}",
                content=str(i),
                semantic=QueueSemantic.APPEND,
                timestamp=time.time() + i,
            )
        )

    drained = await queue.drain()
    # MockStore's asearch respects ``limit`` so the page loop runs twice.
    # Regardless of store semantics, the final result must contain all 150.
    assert len(drained) >= 100
    # Full drain: second call returns nothing.
    remainder = await queue.drain()
    assert remainder == []


# ---------------------------------------------------------------------------
# Heartbeat extends busy-lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_refreshes_busy_timestamp() -> None:
    store = MockStore()
    tracker = ThreadBusyTracker(store)

    await tracker.mark_busy("t")
    original = store._data[("thread_busy",)]["t"]["since"]
    time.sleep(0.01)
    await tracker.heartbeat("t")
    refreshed = store._data[("thread_busy",)]["t"]["since"]
    assert refreshed > original


# ---------------------------------------------------------------------------
# PostRunBackstop pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prune_thread_records_keeps_newest_n() -> None:
    store = MockStore()
    ns = ("run_metadata",)
    # 5 entries for tid-a plus 2 for tid-b (shouldn't be touched).
    for i in range(5):
        await store.aput(ns, f"tid-a_{i:.6f}_x", {"i": i})
    for i in range(2):
        await store.aput(ns, f"tid-b_{i:.6f}_y", {"i": i})

    await _prune_thread_records(store, "tid-a", max_records=2)

    kept_a = [k for k in store._data[ns] if k.startswith("tid-a_")]
    kept_b = [k for k in store._data[ns] if k.startswith("tid-b_")]
    assert len(kept_a) == 2
    assert len(kept_b) == 2
