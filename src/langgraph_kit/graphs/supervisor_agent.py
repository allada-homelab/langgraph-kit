"""Supervisor agent that dynamically routes to sub-agents based on capabilities.

The supervisor uses an LLM (or keyword fallback) to decide which registered
agent should handle each request, delegates to it in a sub-thread, and
synthesizes the result.

Graph structure::

    [START] -> route -> delegate -> synthesize -> [END or route]
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import (  # pyright: ignore[reportMissingModuleSource]
    END,
    StateGraph,
)
from langgraph.graph.message import (  # pyright: ignore[reportMissingModuleSource]
    add_messages,
)

from langgraph_kit.core.orchestration.routing import (
    AgentCapability,
    LLMRoutingStrategy,
    RoutingDecision,
)
from langgraph_kit.llm import build_llm
from langgraph_kit.registry import get, get_metadata, list_agents

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict, total=False):
    """State for the supervisor graph."""

    messages: Annotated[list, add_messages]
    plan: str
    delegations: list[dict[str, Any]]
    pending_result: str
    routing_decision: dict[str, Any]


def build_supervisor_agent(
    checkpointer: Any,
    store: Any,  # noqa: ARG001
    *,
    routing_strategy: Any | None = None,
) -> Any:
    """Build the supervisor agent graph.

    Parameters
    ----------
    checkpointer:
        LangGraph checkpointer for persistence.
    store:
        LangGraph store for key-value data.
    routing_strategy:
        Optional routing strategy. Defaults to ``LLMRoutingStrategy``.

    Returns
    -------
    Compiled LangGraph graph.
    """
    llm = build_llm()

    if routing_strategy is None:
        routing_strategy = LLMRoutingStrategy(llm)

    def _get_capabilities() -> list[AgentCapability]:
        """Build capability list from registered agents, excluding the supervisor itself."""
        capabilities = []
        for agent in list_agents():
            if agent["id"] == "supervisor":
                continue
            meta = get_metadata(agent["id"])
            capabilities.append(
                AgentCapability(
                    agent_id=agent["id"],
                    name=agent["name"],
                    description=meta.description,
                    tags=meta.tags,
                )
            )
        return capabilities

    async def route_node(state: dict[str, Any]) -> dict[str, Any]:
        """Decide which agent should handle the request."""
        messages = state.get("messages", [])
        if not messages:
            return {"pending_result": "No messages to process."}

        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        capabilities = _get_capabilities()
        if not capabilities:
            return {"pending_result": "No agents available for delegation."}

        decision: RoutingDecision = await routing_strategy.route(
            content, capabilities, history=messages[:-1]
        )

        return {
            "routing_decision": decision.model_dump(mode="json"),
        }

    async def delegate_node(state: dict[str, Any]) -> dict[str, Any]:
        """Invoke the target agent in a sub-thread."""
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            HumanMessage,
        )

        decision_data = state.get("routing_decision", {})
        target_id = decision_data.get("target_agent_id", "none")
        prompt = decision_data.get("delegated_prompt", "")

        if target_id == "none" or not prompt:
            return {
                "pending_result": decision_data.get("reasoning", "No suitable agent found."),
            }

        try:
            target_graph = get(target_id)
        except KeyError:
            return {"pending_result": f"Agent '{target_id}' not found."}

        # Run in sub-thread
        sub_thread_id = str(uuid4())
        sub_config: dict[str, Any] = {"configurable": {"thread_id": sub_thread_id}}
        sub_input = {"messages": [HumanMessage(content=prompt)]}

        try:
            result = await target_graph.ainvoke(sub_input, config=sub_config)
            last = result["messages"][-1]
            content = last.content if hasattr(last, "content") else str(last)

            delegation_record = {
                "agent_id": target_id,
                "sub_thread_id": sub_thread_id,
                "prompt": prompt,
                "result_preview": content[:200],
            }

            existing = state.get("delegations", [])
            return {
                "pending_result": content,
                "delegations": [*existing, delegation_record],
            }
        except Exception as exc:
            logger.warning("Delegation to %s failed: %s", target_id, exc)
            return {"pending_result": f"Delegation to {target_id} failed: {exc}"}

    async def synthesize_node(state: dict[str, Any]) -> dict[str, Any]:
        """Synthesize the delegation result into a response."""
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
        )

        pending = state.get("pending_result", "")
        if not pending:
            return {"messages": [AIMessage(content="I couldn't find a suitable response.")]}

        # For now, pass through the delegation result directly.
        # A more sophisticated version would use the LLM to synthesize.
        return {"messages": [AIMessage(content=pending)]}

    # Build the graph
    graph = StateGraph(SupervisorState)
    graph.add_node("route", route_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "delegate")
    graph.add_edge("delegate", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)
