"""Unit tests for ``bind_kit_defaults`` — the compiled-graph default binder.

Regression coverage for a subtle LangChain/LangGraph config-merge
interaction: ``CompiledStateGraph.with_config({"recursion_limit": N})`` is
honored by ``ainvoke`` and ``astream`` (whose ``Pregel.astream`` runs its
own variadic ``ensure_config(self.config, config)`` merge) but **not** by
``astream_events``. That path goes through ``Runnable.astream_events`` →
``langchain_core.tracers.event_stream._astream_events_implementation_v2``,
which calls ``langchain_core.runnables.config.ensure_config(config)`` and
materializes a default ``recursion_limit=25`` into the dict before
dispatching to ``Pregel.astream``. Pregel's merge then treats the 25 as an
explicit caller value and clobbers the bound default.

``bind_kit_defaults`` patches ``astream_events`` on the compiled graph to
pre-merge ``self.config`` into the caller's config before the
langchain-core defaults machinery can fill in its own. These tests verify
that the runtime config seen inside the graph reflects the bound
``recursion_limit`` across every call style.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langchain_core.runnables.config import (  # pyright: ignore[reportMissingModuleSource]
    var_child_runnable_config,
)
from langgraph.graph import (  # pyright: ignore[reportMissingImports]
    END,
    START,
    StateGraph,
)

from langgraph_kit.graphs._builder import bind_kit_defaults


class _S(TypedDict):
    seen_recursion_limit: int


def _build_probe_graph() -> Any:
    """Return a trivial compiled graph whose single node records the runtime
    ``recursion_limit`` from ``var_child_runnable_config``.

    We need a direct probe (not ``GraphRecursionError`` counting) because
    the full deep-agent middleware stack consumes an indeterminate number
    of supersteps, making a raises/doesn't-raise signal flaky for values
    where the bug vs. the fix would differ only by a handful of supersteps.
    """

    def _probe(state: _S) -> _S:
        cfg = var_child_runnable_config.get()
        return {
            "seen_recursion_limit": int(cfg.get("recursion_limit", -1)) if cfg else -1
        }

    g: StateGraph = StateGraph(_S)
    g.add_node("probe", _probe)
    g.add_edge(START, "probe")
    g.add_edge("probe", END)
    return g.compile()


async def _drain_astream_events(graph: Any, input_data: _S, **kwargs: Any) -> _S:
    """Run ``astream_events`` to exhaustion and return the final state."""
    async for ev in graph.astream_events(input_data, version="v2", **kwargs):
        if ev.get("event") == "on_chain_end" and ev.get("name") == "LangGraph":
            data = ev.get("data") or {}
            output = data.get("output")
            if isinstance(output, dict) and "seen_recursion_limit" in output:
                return output  # pyright: ignore[reportReturnType]
    msg = "astream_events did not yield a terminal LangGraph on_chain_end event"
    raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_bound_recursion_limit_reaches_ainvoke() -> None:
    """Sanity: ``ainvoke`` already honored ``with_config``; bind_kit_defaults must not regress it."""
    graph = bind_kit_defaults(_build_probe_graph(), recursion_limit=500)

    result = await graph.ainvoke({"seen_recursion_limit": 0})
    assert result["seen_recursion_limit"] == 500


@pytest.mark.asyncio
async def test_bound_recursion_limit_reaches_astream_events_with_no_config() -> None:
    """The primary regression case — no caller config, bound default must survive.

    Without ``bind_kit_defaults``, ``Runnable.astream_events`` would
    materialize ``recursion_limit=25`` as a default, and the node would
    see 25 instead of 500.
    """
    graph = bind_kit_defaults(_build_probe_graph(), recursion_limit=500)

    result = await _drain_astream_events(graph, {"seen_recursion_limit": 0})
    assert result["seen_recursion_limit"] == 500


@pytest.mark.asyncio
async def test_bound_recursion_limit_reaches_astream_events_with_thread_id_only() -> (
    None
):
    """The arr-assistant chat-route shape: only ``configurable.thread_id`` in config.

    This is exactly the config shape that let the bug ship to production
    before the fix — the caller didn't specify ``recursion_limit``, so
    langchain-core's ``ensure_config`` injected 25 and clobbered the
    bound 500.
    """
    graph = bind_kit_defaults(_build_probe_graph(), recursion_limit=500)

    result = await _drain_astream_events(
        graph,
        {"seen_recursion_limit": 0},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert result["seen_recursion_limit"] == 500


@pytest.mark.asyncio
async def test_caller_recursion_limit_still_overrides_bound_default() -> None:
    """Caller-supplied ``recursion_limit`` must still win over the bound default.

    Mirrors ``ainvoke`` semantics: build-time is a default, runtime
    overrides. ``ensure_config`` applies configs in order with later
    values taking precedence, so ``{bound, caller}`` produces caller.
    """
    graph = bind_kit_defaults(_build_probe_graph(), recursion_limit=500)

    result = await _drain_astream_events(
        graph,
        {"seen_recursion_limit": 0},
        config={"recursion_limit": 77},
    )
    assert result["seen_recursion_limit"] == 77


@pytest.mark.asyncio
async def test_unpatched_graph_confirms_the_bug_this_helper_fixes() -> None:
    """Document the langchain-core bug ``bind_kit_defaults`` works around.

    This test deliberately skips ``bind_kit_defaults`` and uses a bare
    ``with_config({"recursion_limit": 500})``. It asserts the buggy
    behavior on ``astream_events`` so future upstream fixes or behavior
    changes surface immediately as a test failure — at which point the
    workaround in ``bind_kit_defaults`` can be simplified or removed.
    """
    graph = _build_probe_graph().with_config({"recursion_limit": 500})

    # ``ainvoke`` works (this is not the bug) ...
    result_invoke = await graph.ainvoke({"seen_recursion_limit": 0})
    assert result_invoke["seen_recursion_limit"] == 500

    # ... but ``astream_events`` silently drops the bound default.
    result_events = await _drain_astream_events(graph, {"seen_recursion_limit": 0})
    assert result_events["seen_recursion_limit"] == 25, (
        "Expected langchain-core to clobber the bound recursion_limit with its "
        "default of 25 on the astream_events path. If this assertion now sees "
        "500, langchain-core or langgraph has likely fixed the upstream "
        "config-merge bug and the bind_kit_defaults workaround can be "
        "re-evaluated."
    )
