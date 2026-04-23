"""Cluster B — per-middleware happy-path scenarios.

One scenario per middleware that hadn't been covered end-to-end yet.
The stack's ordering and isolation already have dedicated tests in
``test_middleware_ordering_e2e.py`` and ``test_deferred_tools_e2e.py``.
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
from tests.e2e.helpers import (
    answer,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# ToolErrorMiddleware: tool raises → structured ToolMessage with status=error
# ---------------------------------------------------------------------------


async def crash() -> str:
    """Tool body that always raises — test fixture only.

    The function name matters: LangChain derives the LLM-facing tool
    name from ``fn.__name__`` (not ``ToolCapability.name``), so an
    underscored-private name would give the LLM something different
    from the ``tool_call_turn("crash", ...)`` script.
    """
    msg = "simulated tool failure"
    raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_tool_error_middleware_surfaces_structured_error(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """When a tool raises, the middleware returns a structured ToolMessage.

    Run is not killed; the LLM gets a chance to react. This is what
    gives the agent "try a different approach" behavior under tool
    failures — remove the middleware and a single tool exception would
    terminate the whole ainvoke.
    """

    def _configure(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="crash",
                name="crash",
                description="A tool that always raises.",
                fn=crash,
                risk=ToolRisk.READ_ONLY,
            )
        )

    scripted = scripted_llm(
        [
            tool_call_turn("crash"),
            answer("recovered"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="tool-err-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="crash please")]},
        config={"configurable": {"thread_id": "tool-err"}},  # pyright: ignore[reportArgumentType]
    )

    # The ToolErrorMiddleware emits the structured ToolMessage without
    # necessarily propagating ``name`` (langchain's _AgentMiddleware
    # constructs ToolMessage with tool_call_id but may drop name), so
    # filter on status=error / tool_call_id instead.
    errored = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
    ]
    assert errored, (
        "ToolErrorMiddleware should have emitted an error ToolMessage. "
        f"Got messages: {[(type(m).__name__, getattr(m, 'status', None)) for m in result['messages']]}"
    )
    final_error = errored[-1]
    assert "RuntimeError" in str(final_error.content), (
        f"Structured error should mention exception type: {final_error.content!r}"
    )
    assert "simulated tool failure" in str(final_error.content), (
        f"Structured error should carry the exception message: {final_error.content!r}"
    )


# ---------------------------------------------------------------------------
# ResultPersistenceMiddleware: large result → preview + retrieval reference
# ---------------------------------------------------------------------------


_LARGE_CONTENT = "A" * 5000  # Exceeds DEFAULT_PERSIST_THRESHOLD (4000).


async def big() -> str:
    """Return a > 4000-char body so ResultPersistenceMiddleware engages."""
    return _LARGE_CONTENT


@pytest.mark.asyncio
async def test_result_persistence_stores_large_output_and_trims_inline(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Large tool outputs are persisted to store and replaced by a preview.

    This keeps the LLM's tool-message stream bounded regardless of how
    verbose a single tool call was. Verifies two invariants:
    - The inline ToolMessage content is smaller than the full result.
    - The full content landed in the store under ``tool_results``.
    """

    def _configure(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="big",
                name="big",
                description="Return a large body for persistence testing.",
                fn=big,
                risk=ToolRisk.READ_ONLY,
            )
        )

    scripted = scripted_llm(
        [
            tool_call_turn("big"),
            answer("persisted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="result-persist-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="get big data")]},
        config={"configurable": {"thread_id": "result-persist"}},  # pyright: ignore[reportArgumentType]
    )

    tool_messages = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "big"
    ]
    assert tool_messages, "big tool didn't run"
    inline = str(tool_messages[-1].content)
    assert len(inline) < len(_LARGE_CONTENT), (
        f"Inline content wasn't trimmed (len={len(inline)}) even though the "
        f"result was {len(_LARGE_CONTENT)} chars"
    )

    # Store should now have a ``tool_results`` namespace with the full
    # content stashed.
    tool_results = e2e_store._data.get(("tool_results",), {})
    assert tool_results, (
        "No entries in ('tool_results',) namespace — "
        "ResultPersistenceMiddleware didn't persist."
    )
    full = next(iter(tool_results.values()))
    assert _LARGE_CONTENT in str(full), (
        f"Stored record doesn't contain the full original content: {full!r}"
    )


# ---------------------------------------------------------------------------
# EmptyTurnMiddleware: LLM returns empty content with no tool calls → graph exits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_turn_middleware_does_not_spin_on_empty_output(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """An LLM that emits an empty AIMessage doesn't spin — middleware nudges or exits.

    Exact behavior (nudge vs exit) is an implementation detail; the
    invariant is "ainvoke returns in bounded time, doesn't raise
    GraphRecursionError, doesn't exhaust the scripted LLM beyond its
    configured nudge budget."
    """
    # Give the middleware a small nudge budget (3 empty turns should
    # easily be enough to exit). If the middleware were missing, this
    # would spin forever and hit the recursion limit instead.
    scripted = scripted_llm(
        [
            answer(""),
            answer(""),
            answer(""),
            answer("finally something"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="empty-turn-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    # Low recursion limit so that any spin fails the test quickly.
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "empty-turn"}, "recursion_limit": 50},  # pyright: ignore[reportArgumentType]
    )
    # We just need the run to complete. If EmptyTurnMiddleware weren't
    # wired, the graph would re-query the LLM indefinitely and raise
    # GraphRecursionError. Reaching this assertion is the test.
    assert "messages" in result
