"""Routing strategies for the supervisor/router agent."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from langgraph_kit.core.internal_tags import (
    AGENT_ROUTING_TAG,
    internal_llm_config,
)

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
        super().__init__()
        self._llm = llm

    async def route(
        self,
        message: str,
        capabilities: list[AgentCapability],
        history: list[Any] | None = None,
    ) -> RoutingDecision:
        """Decide which agent should handle the request."""
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            BaseMessage,
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

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history[-4:])  # Include recent context
        messages.append(HumanMessage(content=message))

        try:
            # Tag the call so consumers streaming via astream_events can filter
            # the router's chat_model events out of the user-facing transcript
            # — see langgraph_kit.core.internal_tags for rationale.
            response = await self._llm.ainvoke(
                messages,
                config=internal_llm_config(AGENT_ROUTING_TAG, run_name="agent_routing"),
            )
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Tolerant JSON extraction — the LLM sometimes prepends prose
            # or wraps the object in a code fence. A bracket-balanced scan
            # pulls the first top-level ``{...}`` out of the text; if
            # nothing parses, we fall back to "no decision" rather than
            # silently routing to the alphabetically-first agent.
            data = _extract_json_object(content)
            if data is None:
                logger.debug("LLM routing produced unparsable JSON; returning 'none'")
                return RoutingDecision(
                    target_agent_id="none",
                    reasoning="LLM routing produced unparsable JSON",
                    delegated_prompt=message,
                )
            return RoutingDecision(
                target_agent_id=data.get("target_agent_id", "none"),
                reasoning=data.get("reasoning", ""),
                delegated_prompt=data.get("delegated_prompt", message),
            )
        except Exception:
            logger.debug("LLM routing failed", exc_info=True)
            return RoutingDecision(
                target_agent_id="none",
                reasoning="Fallback: LLM routing failed",
                delegated_prompt=message,
            )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from ``text``, tolerating prose and code fences.

    Order of attempts: direct ``json.loads``, strip ```` ``` ```` /
    ```` ```json ```` fences, then bracket-balanced scan for the first
    top-level ``{...}`` substring. Returns ``None`` on total failure
    (caller decides the fallback behaviour).
    """
    import json
    import re as _re

    candidates: list[str] = [text.strip()]
    unfenced = _re.sub(
        r"^```(?:json)?\s*\n?|\n?```\s*$", "", text.strip(), flags=_re.MULTILINE
    )
    if unfenced != text.strip():
        candidates.append(unfenced)
    scanned = _first_balanced_object(text)
    if scanned:
        candidates.append(scanned)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_balanced_object(text: str) -> str | None:
    """Bracket-balanced scan for the first top-level ``{...}`` in ``text``."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "been",
        "they",
        "them",
        "does",
        "also",
        "into",
        "each",
        "over",
        "some",
        "when",
        "your",
        "used",
        "what",
        "which",
        "about",
        "their",
        "would",
        "there",
        "should",
        "other",
        "more",
        "than",
        "could",
        "like",
        "make",
        "just",
        "only",
        "very",
        "most",
    }
)


class KeywordRoutingStrategy:
    """Route requests using keyword matching against agent names, tags, and descriptions.

    No LLM call required — useful for testing and low-latency scenarios.
    Uses word-set matching (not substring) and weighted scoring:
    name words (+3), tags (+2), description words (+1, with stopword filtering).
    """

    async def route(
        self,
        message: str,
        capabilities: list[AgentCapability],
        history: list[Any] | None = None,  # noqa: ARG002
    ) -> RoutingDecision:
        """Match message keywords against agent names, tags, and descriptions."""
        message_words = set(message.lower().split())
        best_score = 0
        best_agent: AgentCapability | None = None

        for cap in capabilities:
            score = 0
            # Score based on agent name words (highest weight)
            for word in (
                cap.agent_id.lower().replace("-", " ").replace("_", " ").split()
            ):
                if len(word) > 2 and word in message_words:
                    score += 3
            # Score based on tag matches (word-set, not substring)
            for tag in cap.tags:
                if tag.lower() in message_words:
                    score += 2
            # Score based on description word matches (with stopword filter)
            for word in cap.description.lower().split():
                if len(word) > 2 and word not in _STOPWORDS and word in message_words:
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
