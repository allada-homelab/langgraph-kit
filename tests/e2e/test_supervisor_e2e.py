"""Cluster J — supervisor agent dispatches to a registered sub-agent.

``build_supervisor_agent`` produces a graph that routes each incoming
message to one of the registered agents, delegates via the chosen
agent's graph in a sub-thread, and synthesizes the reply.

The e2e path end-to-end: register a worker agent, build the supervisor
with a deterministic keyword routing strategy, invoke the supervisor,
and assert the worker's scripted response propagates back up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.orchestration.routing import KeywordRoutingStrategy
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.graphs.supervisor_agent import build_supervisor_agent
from langgraph_kit.registry import AgentMetadata, register
from tests.e2e.helpers import answer, scripted_llm

pytestmark = pytest.mark.e2e


@pytest.fixture
def registered_worker(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> Iterator[str]:
    """Register a minimal worker agent that the supervisor can delegate to."""
    agent_id = "weather-specialist"
    scripted = scripted_llm([answer("sunny and 72")])
    with patched_build_llm(scripted):
        graph, dispatcher = build_deep_agent(
            agent_name=agent_id,
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    register(
        agent_id,
        graph,
        command_dispatcher=dispatcher,
        metadata=AgentMetadata(
            description="Reports weather forecasts",
            tags=["weather", "forecast"],
        ),
    )
    try:
        yield agent_id
    finally:
        from langgraph_kit import registry as registry_mod

        registry_mod._registry.pop(agent_id, None)
        registry_mod._dispatchers.pop(agent_id, None)
        registry_mod._metadata.pop(agent_id, None)


@pytest.mark.asyncio
async def test_supervisor_routes_to_registered_worker_and_echoes_response(
    registered_worker: str,
    checkpointer: Any,
    e2e_store: Any,
) -> None:
    """Supervisor → keyword route → worker graph → synthesized response.

    Uses ``KeywordRoutingStrategy`` so we don't need to script a routing
    LLM on top of the worker's scripted model. The message contains
    "weather" and the registered worker's ``tags=["weather"]`` gives it
    the highest keyword score.
    """
    # The supervisor itself calls build_llm() even though we're using a
    # keyword router. Patch it to a throwaway model — any LLM that
    # satisfies BaseChatModel works; the keyword router never calls it.
    from unittest.mock import patch as _patch

    # Use a scripted LLM with no interactions — it would raise
    # ReplayMismatchError if actually called, which doubles as a guard
    # that keyword routing truly skips LLM calls.
    never_called = scripted_llm([])
    with _patch(
        "langgraph_kit.graphs.supervisor_agent.build_llm",
        return_value=never_called,
    ):
        supervisor = build_supervisor_agent(
            checkpointer=checkpointer,
            store=e2e_store,
            routing_strategy=KeywordRoutingStrategy(),
        )

    result = await supervisor.ainvoke(
        {"messages": [HumanMessage(content="what is the weather today")]},
        config={"configurable": {"thread_id": "sup-1"}},  # pyright: ignore[reportArgumentType]
    )

    # The synthesize node appends an AIMessage with the delegation's
    # pending_result. That result is the worker's final AI content.
    final_contents = [
        str(getattr(m, "content", ""))
        for m in result.get("messages", [])
        if getattr(m, "type", None) == "ai"
    ]
    assert any("sunny and 72" in c for c in final_contents), (
        f"Supervisor didn't forward worker's response; final AI contents: {final_contents!r}"
    )

    # Delegation record was captured.
    delegations = result.get("delegations", [])
    assert delegations, "Supervisor should record the delegation for auditability"
    assert delegations[0]["agent_id"] == registered_worker, (
        f"Delegation should name the routed agent; got {delegations[0]!r}"
    )


@pytest.mark.asyncio
async def test_supervisor_reports_when_no_agent_is_suitable(
    checkpointer: Any,
    e2e_store: Any,
) -> None:
    """With no registered agents, the supervisor returns a graceful message."""
    from unittest.mock import patch as _patch

    never_called = scripted_llm([])
    with _patch(
        "langgraph_kit.graphs.supervisor_agent.build_llm",
        return_value=never_called,
    ):
        supervisor = build_supervisor_agent(
            checkpointer=checkpointer,
            store=e2e_store,
            routing_strategy=KeywordRoutingStrategy(),
        )

    result = await supervisor.ainvoke(
        {"messages": [HumanMessage(content="anything")]},
        config={"configurable": {"thread_id": "sup-empty"}},  # pyright: ignore[reportArgumentType]
    )
    final = next(
        (
            str(getattr(m, "content", ""))
            for m in reversed(result["messages"])
            if getattr(m, "type", None) == "ai"
        ),
        "",
    )
    assert final, "Supervisor should produce some AI response even with no workers"
    # Either "no agents available" or "No suitable response" — the
    # exact text varies but the tone is graceful.
    assert any(
        keyword in final.lower()
        for keyword in ("no agents available", "no suitable", "couldn't find")
    ), f"Supervisor should report the no-workers state gracefully; got {final!r}"
