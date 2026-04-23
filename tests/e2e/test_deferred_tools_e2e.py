"""End-to-end scenarios for the deferred-tools surface.

These are the direct regression guard for the bug that motivated the
whole e2e layer: the kit told the LLM to "call ``tool_search`` first"
in its system prompt while the ``DeferredToolRegistry`` was empty,
producing an always-empty search and (on recursion-bound runs) a spin
on ``tool_search``. The unit tests never caught it because they mocked
``create_deep_agent`` — nothing actually ran the graph with an LLM in
the loop and no test ever examined the prompt the LLM received
relative to the registry state.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    ToolMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    capturing_scripted_llm,
    last_ai_message,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# 1. Happy path — deferred tool discoverable + callable through the dispatcher
# ---------------------------------------------------------------------------


async def _greet(name: str) -> str:
    """Deferred tool body: uppercases the greeting so the output is distinctive."""
    return f"HELLO {name.upper()}"


def _populate_greet(registry: Any) -> None:
    registry.register(
        ToolCapability(
            id="greet",
            name="greet",
            description="Greet a user by name (uppercase).",
            fn=_greet,
            risk=ToolRisk.READ_ONLY,
        )
    )


@pytest.mark.asyncio
async def test_deferred_tool_discoverable_and_callable(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """tool_search → call_deferred_tool → result reaches the LLM as a ToolMessage.

    Populated registry, realistic 3-turn flow:
    - Turn 1: the LLM calls ``tool_search("greet")``
    - Turn 2: the LLM calls ``call_deferred_tool(tool_id="greet", arguments={"name": "Alice"})``
    - Turn 3: the LLM returns a final answer echoing the tool output

    Regression guard: if the dispatcher wiring breaks (lookup by id,
    argument unpacking, async/sync handling, result return), the
    ToolMessage for ``call_deferred_tool`` is either missing or
    contains an error string — this test fails loudly either way.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("tool_search", {"query": "greet"}),
            tool_call_turn(
                "call_deferred_tool",
                {"tool_id": "greet", "arguments": {"name": "Alice"}},
            ),
            answer("Tool replied: HELLO ALICE"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="deferred-e2e-happy",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_deferred_tools=_populate_greet,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="greet Alice")]},
        config={"configurable": {"thread_id": "deferred-happy"}},  # pyright: ignore[reportArgumentType]
    )

    search_msg = assert_tool_invoked(result, "tool_search")
    assert "greet" in search_msg.content, (
        f"tool_search should have returned the deferred tool entry; got {search_msg.content!r}"
    )

    dispatched = assert_tool_invoked(result, "call_deferred_tool")
    assert "HELLO ALICE" in dispatched.content, (
        f"call_deferred_tool should have run greet and returned its output; "
        f"got {dispatched.content!r}"
    )

    final = last_ai_message(result)
    assert "HELLO ALICE" in final.content


# ---------------------------------------------------------------------------
# 2. Empty registry — prompt does NOT push the LLM toward tool_search
# ---------------------------------------------------------------------------


# The distinctive phrase from deferred_tools_awareness in
# src/langgraph_kit/core/prompt_assembly/activation.py. If this string
# appears in the system prompt, the LLM is being instructed to call
# tool_search — which against an empty registry is the exact misdirection
# that motivated this whole e2e layer.
_DEFERRED_AWARENESS_MARKER = "use the tool_search tool to discover"


@pytest.mark.asyncio
async def test_empty_deferred_registry_does_not_push_llm_toward_tool_search(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Default build with an empty DeferredToolRegistry must NOT tell the LLM to tool_search.

    The regression: prompt said "don't assume unavailable — search first"
    while the registry was empty. Fix: ``build_deep_agent`` now auto-gates
    the ``deferred_tools`` condition on ``bool(deferred_registry)``.

    Unit test at the builder level already asserts the composed prompt
    is scrubbed. This test is the e2e counterpart: build a real graph,
    actually invoke it, and check the system message the LLM *received*
    during that invocation via a capturing scripted model. Stronger
    than inspecting compose-time output because it proves the prompt
    survives all middleware transformations on the way to the model.
    """
    capturing = capturing_scripted_llm([answer("hi there")])
    with patched_build_llm(capturing):
        graph, _ = build_reference_deep_agent(
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "deferred-empty"}},  # pyright: ignore[reportArgumentType]
    )

    # Capturing model records every call's input messages. The first
    # call is the one that carries the agent's composed system prompt
    # (later calls may come from sub-workers like the memory extractor).
    assert capturing.captured_calls, "scripted model was never invoked"
    agent_messages = capturing.captured_calls[0]
    combined = "\n".join(
        str(getattr(m, "content", ""))
        for m in agent_messages
    )
    assert _DEFERRED_AWARENESS_MARKER not in combined, (
        "deferred_tools_awareness section leaked into the prompt even though the "
        "DeferredToolRegistry was empty — the regression has returned. "
        f"Prompt excerpt: {combined[:500]!r}"
    )


# ---------------------------------------------------------------------------
# 3. Loop guard — repeated tool_search calls trigger the advisory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_loop_guard_advises_when_llm_spins_on_tool_search(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """After 5+ consecutive tool_search calls, advisory text appears in a ToolMessage.

    ``ToolLoopGuardMiddleware`` is the soft backstop for the class of
    bugs the deferred-tools regression exemplifies: even if the prompt
    accidentally pushes the LLM toward a fruitless loop (or the LLM
    gets stuck on its own), the guard appends a nudge to the 5th+
    tool_search result so the model can notice the pattern.

    This test runs a real graph, scripts 6 consecutive tool_search
    turns, and asserts the guard's advice template actually reached
    the message stream.
    """
    # Populate the deferred registry so tool_search doesn't short-circuit
    # on an empty catalog — we want the flow to reach the guard cleanly.
    # A single entry is enough; the guard doesn't care about results.
    turns = [tool_call_turn("tool_search", {"query": "greet"}) for _ in range(6)]
    turns.append(answer("giving up"))
    scripted = scripted_llm(turns)

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="deferred-e2e-loop",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_deferred_tools=_populate_greet,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="search")]},
        config={"configurable": {"thread_id": "deferred-loop"}},  # pyright: ignore[reportArgumentType]
    )

    search_results = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and m.name == "tool_search"
    ]
    assert len(search_results) >= 5, (
        f"Expected at least 5 tool_search ToolMessages; got {len(search_results)}"
    )
    advisories = [
        m
        for m in search_results
        if "times in a row" in str(m.content) and "loop" in str(m.content).lower()
    ]
    assert advisories, (
        "ToolLoopGuardMiddleware never appended its advisory to a tool_search "
        "result even though the LLM spun 6 times. Contents seen:\n"
        + "\n---\n".join(str(m.content)[:200] for m in search_results)
    )
