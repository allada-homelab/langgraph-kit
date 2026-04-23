"""Cluster H — recursion-limit behavior: build-time default + runtime override.

The kit defaults to ``recursion_limit=100`` (up from LangGraph's 25)
because the full middleware stack burns through supersteps quickly.
This file verifies that (1) a build-time override is honored, (2) a
runtime-time override wins over build, (3) exhaustion raises
``GraphRecursionError`` cleanly rather than silently truncating.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT, build_deep_agent
from tests.e2e.helpers import answer, scripted_llm, tool_call_turn

pytestmark = pytest.mark.e2e


def _always_tool_call_script(n: int) -> list[dict[str, Any]]:
    """Build ``n`` consecutive list_memories calls followed by a final answer.

    list_memories has no args and never fails — ideal for artificially
    inflating the superstep count until the recursion limit is hit.
    """
    return [tool_call_turn("list_memories", {"scope": "user"}) for _ in range(n)] + [
        answer("done")
    ]


@pytest.mark.asyncio
async def test_build_time_recursion_limit_applies(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Passing ``recursion_limit=...`` to ``build_deep_agent`` binds that limit.

    Verified by picking a limit so low it fires on a short script.
    ``GraphRecursionError`` from langgraph is raised cleanly to the
    caller — no partial state corruption, no silent truncation.
    """
    from langgraph.errors import (
        GraphRecursionError,  # pyright: ignore[reportMissingImports]
    )

    # Script many tool calls so the agent is guaranteed to exceed a
    # small build-time limit.
    scripted = scripted_llm(_always_tool_call_script(30))

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="recursion-build-low",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            recursion_limit=5,  # very low — will hit quickly
        )

    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="loop")]},
            config={"configurable": {"thread_id": "recursion-build-low"}},  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.asyncio
async def test_runtime_recursion_limit_wins_over_build_default(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A runtime ``config={"recursion_limit": N}`` must override build-time.

    Build the graph at the kit default (100), pass a low runtime limit
    (5), and confirm the run raises — proving runtime config takes
    precedence over whatever was bound at build.
    """
    from langgraph.errors import (
        GraphRecursionError,  # pyright: ignore[reportMissingImports]
    )

    scripted = scripted_llm(_always_tool_call_script(30))

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="recursion-runtime-low",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            # Default build-time limit (high) — but we'll override at runtime.
        )

    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="loop")]},
            config={  # pyright: ignore[reportArgumentType]
                "configurable": {"thread_id": "recursion-runtime-low"},
                "recursion_limit": 5,
            },
        )


@pytest.mark.asyncio
async def test_short_script_completes_under_default_limit(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Sanity: a short script must complete under the default 100-step limit.

    Guards against regressions that would inadvertently lower the
    default — any realistic run (3 tool calls + final answer) has to
    fit comfortably.
    """
    scripted = scripted_llm(_always_tool_call_script(3))

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="recursion-happy",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="do stuff")]},
        config={"configurable": {"thread_id": "recursion-happy"}},  # pyright: ignore[reportArgumentType]
    )
    assert "messages" in result  # Completed.


def test_default_recursion_limit_is_100() -> None:
    """Documented default is part of the API contract.

    Downstream apps (arr-assistant) rely on this value. A future change
    that lowers it needs to be deliberate.
    """
    assert DEFAULT_RECURSION_LIMIT == 100


# Note: tight regression coverage for the ``astream_events`` codepath —
# where build-time ``recursion_limit`` bindings were silently clobbered to
# 25 by langchain-core's default-materializing ``ensure_config`` — lives in
# ``tests/test_bind_kit_defaults.py``. Those tests observe the runtime
# config directly from inside the graph, which is a tighter signal than
# whether a full deep-agent run raises ``GraphRecursionError`` (the
# middleware stack's superstep budget makes raise/doesn't-raise a flaky
# discriminator for values near 25).
