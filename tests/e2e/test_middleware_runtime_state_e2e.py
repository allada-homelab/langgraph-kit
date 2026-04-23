"""Cluster B edges — ``RuntimeStateMiddleware`` + ``PostRunBackstopMiddleware``.

Both middlewares are always wired via ``build_middleware_stack``. They
don't have dramatic visible outputs like ``CommandMiddleware`` or
``ToolLoopGuard``, so they've historically only had unit coverage.
These tests run a real graph and assert:

- ``RuntimeStateMiddleware`` doesn't crash the run when the agent turn
  succeeds, fails on a tool error, etc. (smoke — the middleware uses
  contextvars and is mostly side-effect-free in terms of state).
- ``PostRunBackstopMiddleware`` persists a ``run_metadata`` record in
  the store after a successful run, with the expected shape.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.resilience.post_run import RUN_METADATA_NAMESPACE
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, scripted_llm

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_runtime_state_middleware_run_completes_cleanly(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Smoke: a normal ainvoke doesn't trip RuntimeStateMiddleware.

    The middleware sets contextvars on before/after hooks. The invariant
    here is just "graph runs without raising" — we already have unit
    tests at ``test_reference_deep_agent`` that verify the contextvar
    values themselves. What this test would catch: a regression where
    the middleware's wrap_model_call interferes with the final-answer
    path and the run hangs or errors.
    """
    scripted = scripted_llm([answer("done")])
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="runtime-state-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "runtime-state"}},  # pyright: ignore[reportArgumentType]
    )
    assert "messages" in result
    # Run reached the final-answer path.
    assert any(
        getattr(m, "type", None) == "ai" and "done" in str(getattr(m, "content", ""))
        for m in result["messages"]
    )


@pytest.mark.asyncio
async def test_post_run_backstop_writes_run_metadata_record(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """After a successful run, a structured record lands in run_metadata.

    The record is keyed ``<thread_id>_<completed_at>`` and contains
    message/tool counts, duration, and a response preview. Regression
    guard: if a future refactor changes where the metadata is persisted
    or what shape it has, downstream observability stops working.
    """
    scripted = scripted_llm([answer("backstop ok")])
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="backstop-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="trigger")]},
        config={"configurable": {"thread_id": "backstop-shape"}},  # pyright: ignore[reportArgumentType]
    )

    records = e2e_store._data.get(RUN_METADATA_NAMESPACE, {})
    assert records, (
        f"PostRunBackstopMiddleware should have written a run_metadata"
        f" record. MockStore namespaces: {list(e2e_store._data.keys())}"
    )
    # Key shape: "<thread_id>_<completed_at_secs>"
    keys = list(records.keys())
    assert any(k.startswith("backstop-shape_") for k in keys), (
        f"Expected record keyed by thread_id prefix; got {keys}"
    )
    record = records[keys[0]]
    # Expected shape — see `_build_run_summary` in
    # src/langgraph_kit/core/resilience/post_run.py.
    for field in (
        "message_count",
        "ai_messages",
        "tool_calls",
        "tool_errors",
        "duration_seconds",
        "completed_at",
        "last_response_preview",
    ):
        assert field in record, (
            f"run_metadata record missing field {field!r}; record keys: {list(record.keys())}"
        )
    assert record["tool_errors"] == 0, (
        f"Happy-path run shouldn't record tool errors; got {record['tool_errors']}"
    )
    assert record["ai_messages"] >= 1, (
        "At least one AIMessage should be counted in the summary"
    )
    assert "backstop ok" in str(record["last_response_preview"]), (
        f"Response preview should carry the final AI content;"
        f" got {record['last_response_preview']!r}"
    )
