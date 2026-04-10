"""Routing strategies for the supervisor/router agent."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentCapability(BaseModel):
    """Describes what a registered agent can do, for routing decisions."""

    agent_id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """The supervisor's routing decision."""

    target_agent_id: str
    reasoning: str = ""
    delegated_prompt: str = ""


class LLMRoutingStrategy:
    """Route requests using an LLM to pick the best agent.

    Presents available agent capabilities to the LLM and asks it to
    choose the best agent and optionally rewrite the prompt for delegation.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def route(
        self,
        message: str,
        capabilities: list[AgentCapability],
        history: list[Any] | None = None,
    ) -> RoutingDecision:
        """Decide which agent should handle the request."""
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            HumanMessage,
            SystemMessage,
        )

        agents_desc = "\n".join(
            f"- **{cap.agent_id}**: {cap.description} (tags: {', '.join(cap.tags)})"
            for cap in capabilities
        )

        system_prompt = (
            "You are a routing supervisor. Given a user request and available agents, "
            "decide which agent should handle it.\n\n"
            f"Available agents:\n{agents_desc}\n\n"
            "Respond in this exact JSON format:\n"
            '{"target_agent_id": "agent-id", "reasoning": "why this agent", '
            '"delegated_prompt": "rewritten prompt for the agent"}\n\n'
            "If none of the agents are suitable, set target_agent_id to 'none'."
        )

        messages = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history[-4:])  # Include recent context
        messages.append(HumanMessage(content=message))

        try:
            response = await self._llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)

            import json

            # Try to parse JSON from the response
            data = json.loads(content)
            return RoutingDecision(
                target_agent_id=data.get("target_agent_id", "none"),
                reasoning=data.get("reasoning", ""),
                delegated_prompt=data.get("delegated_prompt", message),
            )
        except Exception:
            logger.debug("LLM routing failed, using first available agent", exc_info=True)
            if capabilities:
                return RoutingDecision(
                    target_agent_id=capabilities[0].agent_id,
                    reasoning="Fallback: LLM routing failed",
                    delegated_prompt=message,
                )
            return RoutingDecision(
                target_agent_id="none",
                reasoning="No agents available",
                delegated_prompt=message,
            )


class KeywordRoutingStrategy:
    """Route requests using keyword matching against agent tags/descriptions.

    No LLM call required — useful for testing and low-latency scenarios.
    """

    async def route(
        self,
        message: str,
        capabilities: list[AgentCapability],
        history: list[Any] | None = None,  # noqa: ARG002
    ) -> RoutingDecision:
        """Match message keywords against agent tags and descriptions."""
        message_lower = message.lower()
        best_score = 0
        best_agent: AgentCapability | None = None

        for cap in capabilities:
            score = 0
            # Score based on tag matches
            for tag in cap.tags:
                if tag.lower() in message_lower:
                    score += 2
            # Score based on description word matches
            for word in cap.description.lower().split():
                if len(word) > 3 and word in message_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_agent = cap

        if best_agent is not None:
            return RoutingDecision(
                target_agent_id=best_agent.agent_id,
                reasoning=f"Keyword match (score={best_score})",
                delegated_prompt=message,
            )

        # Fallback: first agent
        if capabilities:
            return RoutingDecision(
                target_agent_id=capabilities[0].agent_id,
                reasoning="No keyword match, using default",
                delegated_prompt=message,
            )

        return RoutingDecision(
            target_agent_id="none",
            reasoning="No agents available",
            delegated_prompt=message,
        )
