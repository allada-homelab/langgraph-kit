"""Command interceptor middleware — routes slash-commands before they reach the LLM."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware as _AgentMiddleware,
)

from langgraph_kit.core.commands.dispatch import CommandDispatcher

logger = logging.getLogger(__name__)


class CommandMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Intercepts user messages starting with ``/`` and routes them to the CommandDispatcher.

    If the command is handled, the middleware short-circuits the agent loop
    by injecting the command output as an AI message and skipping the LLM call.
    Unrecognized commands pass through to the agent normally.
    """

    def __init__(self, dispatcher: CommandDispatcher) -> None:
        super().__init__()
        self._dispatcher = dispatcher

    async def abefore_agent(
        self,
        state: Any,
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check the last user message for a slash command."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        if not isinstance(content, str) or not content.strip().startswith("/"):
            return None

        if not self._dispatcher.is_command(content.strip()):
            return None

        # Dispatch the command
        result = await self._dispatcher.dispatch(
            content.strip(),
            context={"messages": messages, "state": state},
        )

        if not result.handled:
            return None

        # Short-circuit: replace the last user message exchange with command output
        from langchain_core.messages import (
            AIMessage,  # pyright: ignore[reportMissingModuleSource]
        )

        # Some commands (e.g. /compact) produce a replacement message list
        compacted: list[Any] | None = result.metadata.get("compacted_messages")
        if compacted is not None:
            compacted.append(AIMessage(content=result.output))
            return {"messages": compacted}

        return {"messages": [AIMessage(content=result.output)]}
