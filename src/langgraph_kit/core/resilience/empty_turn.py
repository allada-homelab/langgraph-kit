"""EmptyTurnMiddleware — prevent the model from ending a turn with no output.

Detects turns where the model produced no meaningful text and no tool calls,
then nudges it to take a concrete next step instead of silently terminating.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_kit.core.resilience._message_text import aimessage_text

logger = logging.getLogger(__name__)

# Minimum content length to count as "meaningful" (strips whitespace)
_MIN_CONTENT_LENGTH = 5


class EmptyTurnMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Inspects model output after each call — injects a nudge if the turn is empty.

    An "empty turn" is one where the model produced:
    - No text content (or only whitespace), AND
    - No tool calls

    When detected, a synthetic user message is appended telling the model
    to take a concrete action or explain why it stopped.
    """

    def __init__(self, *, max_nudges: int = 2) -> None:
        super().__init__()
        self._nudge_count = 0
        self._max_nudges = max_nudges

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        content_str = aimessage_text(last)
        has_content = bool(
            content_str.strip() and len(content_str.strip()) >= _MIN_CONTENT_LENGTH
        )
        has_tool_calls = bool(last.tool_calls)

        if has_content or has_tool_calls:
            # Valid turn — reset nudge counter
            self._nudge_count = 0
            return None

        # Empty turn detected
        self._nudge_count += 1

        if self._nudge_count > self._max_nudges:
            # Too many nudges — let it end to avoid infinite loops
            logger.warning(
                "Empty turn detected but max nudges (%d) exceeded — allowing termination",
                self._max_nudges,
            )
            return None

        logger.info(
            "Empty turn detected (nudge %d/%d)", self._nudge_count, self._max_nudges
        )

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "[System: Your last response was empty — no text and no tool calls. "
                        "Please take a concrete next step: use a tool, provide an answer, "
                        "or explain why no further action is needed.]"
                    )
                )
            ]
        }
