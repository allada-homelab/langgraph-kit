"""Regression tests for ``retrieve_result`` cross-thread isolation.

Before the fix, ``retrieve_result`` looked up refs under a shared
``("tool_results",)`` namespace — any thread that learned a ref hash
could read content written by any other thread. Now the namespace is
``("tool_results", thread_id)`` and refs are thread-local.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.tools.result_retrieval import (
    build_result_retrieval_tool,
    tool_results_namespace,
)

from .conftest import MockStore


def test_tool_results_namespace_is_thread_scoped() -> None:
    assert tool_results_namespace("a") == ("tool_results", "a")
    assert tool_results_namespace("b") == ("tool_results", "b")
    assert tool_results_namespace("a") != tool_results_namespace("b")


@pytest.mark.asyncio
async def test_retrieve_result_cannot_read_other_thread_content(
    monkeypatch: Any,
) -> None:
    """Even knowing the ref, thread B cannot reach thread A's persisted result."""
    store = MockStore()
    await store.aput(
        ("tool_results", "thread-a"),
        "ref-xyz",
        {
            "content": "secret-from-a",
            "tool_name": "t",
            "tool_call_id": "call-1",
            "char_count": 13,
        },
    )

    # Fake graph runtime config pointing at thread-b.
    fake_config = {"configurable": {"thread_id": "thread-b"}}
    monkeypatch.setattr(
        "langgraph.config.get_config",
        lambda: fake_config,
    )

    retrieve = build_result_retrieval_tool(store)
    out = await retrieve(result_ref="ref-xyz")
    assert "secret-from-a" not in out
    assert "No persisted result found" in out


@pytest.mark.asyncio
async def test_retrieve_result_reads_own_thread_content(monkeypatch: Any) -> None:
    store = MockStore()
    await store.aput(
        ("tool_results", "thread-a"),
        "ref-xyz",
        {
            "content": "own-content",
            "tool_name": "t",
            "tool_call_id": "call-1",
            "char_count": 11,
        },
    )

    monkeypatch.setattr(
        "langgraph.config.get_config",
        lambda: {"configurable": {"thread_id": "thread-a"}},
    )

    retrieve = build_result_retrieval_tool(store)
    out = await retrieve(result_ref="ref-xyz")
    assert "own-content" in out


@pytest.mark.asyncio
async def test_retrieve_result_errors_without_runtime_context(
    monkeypatch: Any,
) -> None:
    store = MockStore()

    def _raise() -> dict[str, Any]:
        raise RuntimeError("no config")

    monkeypatch.setattr("langgraph.config.get_config", _raise)

    retrieve = build_result_retrieval_tool(store)
    out = await retrieve(result_ref="ref-xyz")
    assert "requires a graph runtime context" in out


@pytest.mark.asyncio
async def test_retrieve_result_errors_when_thread_id_missing(
    monkeypatch: Any,
) -> None:
    store = MockStore()
    monkeypatch.setattr(
        "langgraph.config.get_config",
        lambda: {"configurable": {}},
    )

    retrieve = build_result_retrieval_tool(store)
    out = await retrieve(result_ref="ref-xyz")
    assert "thread_id missing" in out
