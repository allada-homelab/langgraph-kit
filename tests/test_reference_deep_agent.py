"""Tests for reference_deep_agent build function, RuntimeStateMiddleware, and StopHooksMiddleware."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
from langgraph_kit.core.resilience.runtime_state import RuntimeStateMiddleware
from langgraph_kit.core.resilience.stop_hooks import StopHooksMiddleware
from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent

# ---------------------------------------------------------------------------
# GENERAL_WORKERS tests
# ---------------------------------------------------------------------------


def test_worker_definitions_valid() -> None:
    """GENERAL_WORKERS has 3 entries with name, description, system_prompt."""
    assert len(GENERAL_WORKERS) == 3
    for defn in GENERAL_WORKERS:
        assert "name" in defn
        assert "description" in defn
        assert "system_prompt" in defn
        assert isinstance(defn["name"], str)
        assert defn["name"]
        assert isinstance(defn["description"], str)
        assert defn["description"]
        assert isinstance(defn["system_prompt"], str)
        assert defn["system_prompt"]


# ---------------------------------------------------------------------------
# RuntimeStateMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_state_middleware_tracks_state() -> None:
    """Create RuntimeStateMiddleware, call abefore_agent, verify state='started'."""
    mw = RuntimeStateMiddleware()
    assert mw.state == "idle"

    await mw.abefore_agent({}, MagicMock())
    assert mw.state == "started"
    assert mw.turn_count == 1


@pytest.mark.asyncio
async def test_runtime_state_middleware_model_call_success() -> None:
    """Mock handler, verify state transitions to 'completed'."""
    mw = RuntimeStateMiddleware()
    await mw.abefore_agent({}, MagicMock())

    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = AsyncMock(return_value=mock_response)
    result = await mw.awrap_model_call(mock_request, mock_handler)

    assert result is mock_response
    assert mw.state == "completed"
    assert mw.stop_reason == "final_answer"


@pytest.mark.asyncio
async def test_runtime_state_middleware_model_call_failure() -> None:
    """Mock handler raises, verify state='failed'."""
    mw = RuntimeStateMiddleware()
    await mw.abefore_agent({}, MagicMock())

    mock_request = MagicMock()
    mock_handler = AsyncMock(side_effect=ValueError("something broke"))

    with pytest.raises(ValueError, match="something broke"):
        await mw.awrap_model_call(mock_request, mock_handler)

    assert mw.state == "failed"
    assert mw.stop_reason is not None
    assert "ValueError" in mw.stop_reason


# ---------------------------------------------------------------------------
# StopHooksMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_hooks_middleware_runs_hooks() -> None:
    """Register a mock hook with on_turn_complete, verify it's called."""
    hook = AsyncMock()
    hook.on_turn_complete = AsyncMock()
    hook.blocking = False

    mw = StopHooksMiddleware(hooks=[hook])

    await mw.aafter_agent({"messages": []}, MagicMock())

    hook.on_turn_complete.assert_awaited_once_with({"messages": []})


@pytest.mark.asyncio
async def test_stop_hooks_middleware_non_blocking_failure() -> None:
    """Hook raises, but middleware doesn't crash (non-blocking)."""
    hook = MagicMock()
    hook.on_turn_complete = AsyncMock(side_effect=RuntimeError("hook failed"))
    hook.blocking = False

    mw = StopHooksMiddleware(hooks=[hook])

    # Should not raise
    await mw.aafter_agent({"messages": []}, MagicMock())

    hook.on_turn_complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# build_reference_deep_agent smoke test
# ---------------------------------------------------------------------------


def test_build_reference_deep_agent_returns_graph(mock_store: Any) -> None:
    """Call build_reference_deep_agent with mock store + mock checkpointer."""
    checkpointer = MagicMock()

    fake_graph = MagicMock(name="compiled_graph")
    # `with_config` returns itself so the identity check below continues to work;
    # real `CompiledStateGraph.with_config` returns a new graph with merged config.
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph
    fake_llm = MagicMock(name="fake_llm")

    # Mock deepagents and its backend submodules so lazy imports resolve
    backends_mod = MagicMock()
    module_patches = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }

    with (
        patch.dict(sys.modules, module_patches),
        patch("langgraph_kit.graphs._builder.build_llm", return_value=fake_llm),
    ):
        graph, _dispatcher = build_reference_deep_agent(
            checkpointer=checkpointer, store=mock_store
        )

    assert graph is fake_graph
    deepagents_mod.create_deep_agent.assert_called_once()
    fake_graph.with_config.assert_called_once_with({"recursion_limit": 100})


def test_build_reference_deep_agent_accepts_recursion_limit_override(
    mock_store: Any,
) -> None:
    """Custom recursion_limit is forwarded to the compiled graph."""
    checkpointer = MagicMock()

    fake_graph = MagicMock(name="compiled_graph")
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph
    fake_llm = MagicMock(name="fake_llm")

    backends_mod = MagicMock()
    module_patches = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }

    with (
        patch.dict(sys.modules, module_patches),
        patch("langgraph_kit.graphs._builder.build_llm", return_value=fake_llm),
    ):
        build_reference_deep_agent(
            checkpointer=checkpointer,
            store=mock_store,
            recursion_limit=500,
        )

    fake_graph.with_config.assert_called_once_with({"recursion_limit": 500})
