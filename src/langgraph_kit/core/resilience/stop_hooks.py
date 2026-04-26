"""Stop hooks lifecycle middleware.

Runs registered lifecycle hooks at turn boundaries.
Non-blocking hooks log failures; blocking hooks propagate exceptions.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class StopHooksMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Runs registered lifecycle hooks at turn boundaries.

    Hooks are executed in order after the agent completes.
    Non-blocking hooks log failures; blocking hooks propagate exceptions.
    """

    def __init__(self, hooks: list[Any] | None = None) -> None:
        super().__init__()
        self._hooks: list[Any] = hooks or []

    def register_hook(self, hook: Any) -> None:
        self._hooks.append(hook)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        for hook in self._hooks:
            try:
                if hasattr(hook, "on_turn_complete"):
                    await hook.on_turn_complete(state)
            except Exception:
                blocking = getattr(hook, "blocking", False)
                if blocking:
                    raise
                logger.exception("Non-blocking hook failed: %s", hook)
        return None


class TurnTelemetryStopHook:
    """Default stop hook used by the reference deep agent.

    Emits a single ``logger.debug`` line per turn with the message count
    and the number of tool calls on the final ``AIMessage``. Non-blocking
    by design — exceptions from this hook are logged and swallowed by
    :class:`StopHooksMiddleware` rather than failing the turn.

    Suitable as a baseline observability hook; replace or extend via
    ``extra_stop_hooks=`` on the reference builder for richer telemetry.
    """

    blocking: bool = False

    async def on_turn_complete(self, state: Any) -> None:
        messages: list[Any] = []
        if isinstance(state, dict):
            raw = state.get("messages")
            if isinstance(raw, list):
                messages = raw

        tool_call_count = 0
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage):
                tool_call_count = len(last.tool_calls or [])

        logger.debug(
            "turn_complete messages=%d tool_calls=%d",
            len(messages),
            tool_call_count,
        )
