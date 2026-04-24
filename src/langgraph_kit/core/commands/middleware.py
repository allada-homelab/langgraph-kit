"""Command interceptor middleware — routes slash-commands before they reach the LLM."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware as _AgentMiddleware,
)
from langchain.agents.middleware.types import (
    hook_config,
)

from langgraph_kit.core.commands.dispatch import CommandDispatcher

logger = logging.getLogger(__name__)

# Marker that the streaming layer (``streaming.py``) reads to decide
# whether to emit a ``command_result`` SSE event. Without this marker,
# any run that ended silently (e.g. tool-only deterministic pipelines)
# would get its last AIMessage mis-reported as a command result.
COMMAND_RESULT_MARKER = "_lgkit_command_result"


class CommandMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Intercepts user messages starting with ``/`` and routes them to the CommandDispatcher.

    If the command is handled, the middleware short-circuits the agent loop
    by injecting the command output as an AI message and jumping to ``end``
    so the LLM is never called for this turn. Unrecognized commands pass
    through to the agent normally.

    The short-circuit requires two cooperating pieces that a naive
    implementation would miss: a ``jump_to: "end"`` return value AND the
    :func:`langchain.agents.middleware.types.hook_config` ``can_jump_to``
    allowlist — without the decorator, LangChain's graph wiring ignores
    ``jump_to`` and the model runs anyway, producing a duplicate response.
    """

    def __init__(self, dispatcher: CommandDispatcher) -> None:
        super().__init__()
        self._dispatcher = dispatcher

    @hook_config(can_jump_to=["end"])
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

        # Short-circuit: emit the command result as an AI message and
        # jump_to "end" so the graph terminates without invoking the model.
        # Returning messages goes through the ``add_messages`` reducer which
        # appends by default but *replaces in place* when message ids match
        # — which is how the ``/compact`` branch below delivers a mutated
        # transcript without leaving duplicates.
        from langchain_core.messages import (
            AIMessage,  # pyright: ignore[reportMissingModuleSource]
        )

        marker_kwargs = {COMMAND_RESULT_MARKER: True}
        compacted: list[Any] | None = result.metadata.get("compacted_messages")
        if compacted is not None:
            return {
                "messages": [
                    *compacted,
                    AIMessage(content=result.output, additional_kwargs=marker_kwargs),
                ],
                "jump_to": "end",
            }

        return {
            "messages": [
                AIMessage(content=result.output, additional_kwargs=marker_kwargs)
            ],
            "jump_to": "end",
        }
