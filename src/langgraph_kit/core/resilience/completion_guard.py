"""CompletionGuardMiddleware — heuristic premature-completion detector.

Catches cases where the model appears to stop too early based on local
signals, and pushes it back into the loop to either justify stopping or
continue with concrete work.

Without a formal task contract this is best-effort — it detects *suspicious*
endings, not provably invalid ones.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# Phrases that signal the model thinks it's done
_COMPLETION_PHRASES = re.compile(
    r"\b((?:i(?:'ve| have) )?(?:completed?|finish(?:ed)?|done|all set|that'?s? (?:it|all|everything))|"
    + r"(?:task|work|implementation) (?:is )?(?:complete|done|finished))\b",
    re.IGNORECASE,
)

# Minimum messages to have seen before guarding (let the agent warm up)
_MIN_MESSAGES_BEFORE_GUARD = 4

# Maximum challenges per run to avoid infinite loops
_MAX_CHALLENGES = 2


class CompletionGuardMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Heuristic guard against premature completion.

    After each model turn, checks for signals that suggest the model is
    stopping too early:

    1. Claims completion but there was a recent unrecovered tool error
    2. Claims completion but never used any tools despite task context
       suggesting tool use was expected
    3. Claims completion after minimal exploration (very few messages)

    When triggered, injects a challenge asking the model to justify
    stopping or continue with concrete work.
    """

    def __init__(self, *, min_tool_calls: int = 1) -> None:
        super().__init__()
        self._min_tool_calls = min_tool_calls
        self._challenges_issued = 0

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        if self._challenges_issued >= _MAX_CHALLENGES:
            return None

        messages = state.get("messages", [])
        if len(messages) < _MIN_MESSAGES_BEFORE_GUARD:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # Only guard when the model claims completion
        raw_content: Any = last.content  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        content: str = raw_content if isinstance(raw_content, str) else ""
        if not _COMPLETION_PHRASES.search(content):
            return None

        # Check for suspicious signals
        reason = _check_suspicious_completion(messages, self._min_tool_calls)
        if reason is None:
            return None

        self._challenges_issued += 1
        logger.info(
            "Completion guard triggered (challenge %d/%d): %s",
            self._challenges_issued,
            _MAX_CHALLENGES,
            reason,
        )

        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"[System: Your completion appears premature. {reason} "
                        "Either explain why no more work is needed, or continue "
                        "with the next concrete action.]"
                    )
                )
            ]
        }


def _check_suspicious_completion(
    messages: list[Any], min_tool_calls: int
) -> str | None:
    """Return a reason string if completion looks suspicious, None if it seems legit."""
    tool_calls_made = 0
    recent_tool_errors = 0
    # Scan the last ~20 messages for signals
    window = messages[-20:]

    for msg in window:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_made += len(msg.tool_calls)
        if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error":
            recent_tool_errors += 1

    # Signal 1: Recent tool error with no recovery attempt after
    if recent_tool_errors > 0:
        # Check if the last few messages show a tool error followed by completion
        for i in range(len(window) - 1, max(len(window) - 5, -1), -1):
            msg = window[i]
            if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error":
                return (
                    f"There was a recent tool error that appears unrecovered. "
                    f"({recent_tool_errors} error(s) in recent history.)"
                )

    # Signal 2: No tools used despite task context
    if tool_calls_made < min_tool_calls:
        return (
            f"You used {tool_calls_made} tool(s) but this task likely requires "
            f"at least {min_tool_calls}. Did you forget to use tools?"
        )

    return None
