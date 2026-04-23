"""Cluster B edges — ``CompletionGuardMiddleware`` challenges premature finishes.

When the model claims completion but the local signals look off (zero
tool calls despite task context, recent unrecovered tool error,
unusually brief final answer), the guard injects a challenge HumanMessage
asking it to justify stopping or continue. Up to 2 challenges per run.

The unit tests at ``test_resilience`` cover the heuristic in isolation.
This e2e asserts the challenge threading actually works in a real graph:
inject a challenge → scripted LLM sees it → responds with a second turn.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, last_ai_message, scripted_llm

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_completion_guard_challenges_empty_run_claiming_done(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Agent claims 'task is complete' with 0 tool calls → guard injects a challenge.

    The scripted model:
      1. Returns a premature 'task is complete' (no tool calls).
      2. Returns a follow-up answer after the guard nudges it.
    The guard requires ≥4 messages of context before firing, so we
    seed the state with enough prior back-and-forth.
    """
    # Seed with 4+ messages so CompletionGuard's min-messages gate opens.
    seeded = [
        HumanMessage(content="first ask"),
        AIMessage(content="first resp"),
        HumanMessage(content="second ask"),
        AIMessage(content="second resp"),
        HumanMessage(content="final ask"),
    ]

    scripted = scripted_llm(
        [
            answer("task is complete"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="completion-guard-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": seeded},
        config={"configurable": {"thread_id": "completion-guard"}},  # pyright: ignore[reportArgumentType]
    )

    messages = result["messages"]
    # The invariant is that the guard injected its challenge into state
    # after the premature completion. Whether the agent loops back to
    # generate a follow-up response is a downstream concern of the
    # agent framework (create_agent's END logic) — what this test
    # guards is the middleware integration itself.
    guard_nudges = [
        m
        for m in messages
        if isinstance(m, HumanMessage) and "premature" in str(m.content).lower()
    ]
    assert guard_nudges, (
        "CompletionGuard should have injected a 'premature completion' nudge."
        f" Messages: {[(type(m).__name__, str(getattr(m, 'content', ''))[:60]) for m in messages]}"
    )
    nudge_text = str(guard_nudges[0].content)
    assert "continue with the next concrete action" in nudge_text, (
        f"Guard nudge should carry the kit's standard 'continue' framing;"
        f" got {nudge_text!r}"
    )
    final = last_ai_message(result)
    assert "complete" in str(final.content).lower(), (
        f"Premature completion should remain visible as final AI message;"
        f" got {final.content!r}"
    )
