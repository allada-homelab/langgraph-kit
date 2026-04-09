"""Deepagents middleware that triggers auto memory extraction after eligible turns."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from langgraph_kit.core.memory.extraction import AutoMemoryExtractor
from langgraph_kit.core.memory.models import MemoryScope

logger = logging.getLogger(__name__)

# Tool names that count as "agent wrote memory this turn"
_MEMORY_WRITE_TOOLS = {"save_memory", "update_memory", "delete_memory"}

# Per-request tracking via contextvars (thread/coroutine safe)
_agent_wrote_memory: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_agent_wrote_memory", default=False
)


class ExtractionMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Deepagents middleware that runs auto memory extraction after eligible turns.

    Uses contextvars for per-request state so concurrent invocations don't interfere.
    """

    def __init__(
        self,
        extractor: AutoMemoryExtractor,
        scope: MemoryScope = MemoryScope.USER,
        recent_message_window: int = 10,
    ) -> None:
        super().__init__()
        self._extractor = extractor
        self._scope = scope
        self._window = recent_message_window

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Reset per-turn tracking."""
        _agent_wrote_memory.set(False)
        return None

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Track memory tool usage."""
        result = await handler(request)
        tool_name = (
            request.tool_call.get("name", "") if hasattr(request, "tool_call") else ""
        )
        if tool_name in _MEMORY_WRITE_TOOLS:
            _agent_wrote_memory.set(True)
        return result

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """After agent completes, run extraction on recent messages."""
        try:
            state_dict: dict[str, Any] = dict(state) if isinstance(state, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
            messages_list: list[Any] = state_dict.get("messages", [])
            recent_msgs: list[Any] = (
                messages_list[-self._window :] if messages_list else []
            )
            await self._extractor.extract(
                recent_messages=recent_msgs,
                scope=self._scope,
                agent_wrote_memory_this_turn=_agent_wrote_memory.get(),
            )
        except Exception:
            logger.exception("Post-turn memory extraction failed (non-blocking)")
        return None
