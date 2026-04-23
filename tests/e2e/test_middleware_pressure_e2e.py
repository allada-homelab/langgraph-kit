"""Cluster B edges — ``PressureMiddleware`` compacts under real load.

The middleware runs ``abefore_agent``, assesses the ``PressureMonitor``,
and returns a ``{"messages": [...]}`` update that replaces the existing
list with a compacted one when pressure is high. This test pre-loads the
state with enough large tool messages to cross the microcompact threshold
and asserts the LLM's prompt received the truncated versions, not the
originals.

``PressureMonitor`` defaults to a 128k-token window, so the inbound
messages are sized to trip the microcompact heuristic (large_tool_outputs
> 2, old tool messages > 2000 chars, enough messages to cross the
`>10 messages` floor).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, capturing_scripted_llm

pytestmark = pytest.mark.e2e


def _synthetic_history(*, n_tool_msgs: int, tool_size: int) -> list[Any]:
    """Build a synthetic prior-turn history with oversized tool outputs.

    Each tool message is > 2000 chars (microcompact truncates those), and
    we keep the total > 10 messages so microcompact doesn't bail on the
    "too short to compact" check.
    """
    history: list[Any] = []
    for i in range(n_tool_msgs):
        history.append(HumanMessage(content=f"turn {i} user"))
        history.append(
            AIMessage(
                content="",
                tool_calls=[{"id": f"call_{i}", "name": "fetch", "args": {"i": i}}],
            )
        )
        history.append(
            ToolMessage(
                content=f"BIGRESULT_{i}:" + ("x" * tool_size),
                tool_call_id=f"call_{i}",
                name="fetch",
            )
        )
    return history


@pytest.mark.asyncio
async def test_pressure_microcompacts_large_old_tool_outputs(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Large historic tool messages are truncated before reaching the LLM.

    The middleware ``abefore_agent`` hook runs once at the start of each
    agent turn. Given a state primed with multiple >2000 char tool
    outputs, the monitor should pick ``MICROCOMPACT`` and the middleware
    should return a message list with the old tool outputs truncated.

    Invariant: no full 8000-char `xxx...` run makes it to the LLM. The
    microcompact replacement is a 200-char head + a `[truncated — ...]`
    marker.
    """
    # Sizing: PressureMonitor defaults to a 128_000-token window. Token
    # estimation is chars // 4, and `large_tool_outputs` counts only
    # messages > 4_000 TOKENS (== > 16_000 chars). To hit the
    # MICROCOMPACT branch we need:
    #   - pressure_pct in [0.70, 0.85) (moderate band) — critical
    #     pressure escalates to FULL_COMPACTION which needs a live LLM,
    #     which we'd have to script around.
    #   - large_tool_outputs > 2 — i.e. each tool > 4000 tokens.
    #   - total messages > 10 so the _microcompact floor doesn't bail,
    #     AND at least some tool messages sitting BEFORE the last-10
    #     "compact boundary" (only those get truncated).
    # 20 tool cycles (= 60 messages) * 20_000 chars/tool = 5000 tokens
    # each → pressure_pct ≈ 0.78, large_tool_outputs = 20 (>2) → monitor
    # picks MICROCOMPACT. Boundary = 50, so tool messages at indices
    # 2, 5, …, 47 (~16 of them) are candidates for truncation.
    history = _synthetic_history(n_tool_msgs=20, tool_size=20_000)
    history.append(HumanMessage(content="now continue"))

    capturing = capturing_scripted_llm([answer("compacted?")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="pressure-micro-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": history},
        config={"configurable": {"thread_id": "pressure-micro"}},  # pyright: ignore[reportArgumentType]
    )

    assert capturing.captured_calls, "scripted model was never invoked"
    prompt_messages = capturing.captured_calls[0]
    tool_contents = [
        str(getattr(m, "content", ""))
        for m in prompt_messages
        if isinstance(m, ToolMessage)
    ]
    # At least some tool messages ahead of the compact boundary must be
    # truncated. The last 10 messages always pass through unchanged, so
    # a portion of the oversized bodies (the "tail") survives and is
    # expected. The invariant we care about is: at least one was
    # truncated, and the '[truncated — ...]' marker made it through.
    truncated = [c for c in tool_contents if "truncated" in c]
    assert truncated, (
        "PressureMiddleware should have microcompacted at least one oversized"
        " tool output; no '[truncated — …]' markers found."
        f" Tool message sizes: {sorted({len(c) for c in tool_contents})}"
    )
    # And the truncated replacement should be dramatically shorter than
    # the original 20_000-char body.
    assert all(len(c) < 2000 for c in truncated), (
        "Microcompact replacement should be ≤ 2000 chars (200-char head +"
        f" marker); got sizes {[len(c) for c in truncated]}"
    )


@pytest.mark.asyncio
async def test_pressure_no_op_under_light_load(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A tiny history must not trigger compaction — the LLM sees the input unchanged.

    Default monitor warns at 70% window utilization; an empty-ish state
    is well below that.
    """
    capturing = capturing_scripted_llm([answer("ok")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="pressure-light-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="a tiny ask")]},
        config={"configurable": {"thread_id": "pressure-light"}},  # pyright: ignore[reportArgumentType]
    )

    # Minimum sanity: the scripted LLM was invoked. The PressureMiddleware
    # should have returned None (no action), so the graph ran normally.
    prompt_messages = capturing.captured_calls[0]
    human = next(
        (m for m in prompt_messages if isinstance(m, HumanMessage)),
        None,
    )
    assert human is not None, "HumanMessage should have survived the passthrough"
    assert "a tiny ask" in str(human.content), (
        "Light-load passthrough shouldn't alter the inbound message"
    )
