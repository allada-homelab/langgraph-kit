"""Deepagents middleware for context pressure monitoring and mitigation."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from langgraph_kit.core.context_management.pressure import (
    MitigationStrategy,
    PressureMonitor,
)

logger = logging.getLogger(__name__)


class PressureMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Deepagents middleware that checks context pressure before each agent turn.

    Returns state updates with compacted messages instead of mutating in place.
    """

    def __init__(self, monitor: PressureMonitor) -> None:
        super().__init__()
        self._monitor = monitor

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Check pressure and return compacted messages if needed."""
        state_dict: dict[str, Any] = dict(state) if isinstance(state, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
        messages: list[Any] = state_dict.get("messages", [])
        if not messages:
            return None

        signals = self._monitor.assess(messages)
        strategy = self._monitor.choose_mitigation(signals)

        if strategy == MitigationStrategy.NONE:
            return None

        logger.info(
            "Context pressure: %.0f%% (%d tokens), strategy: %s",
            signals.pressure_pct * 100,
            signals.estimated_tokens,
            strategy.value,
        )

        if strategy == MitigationStrategy.MICROCOMPACT:
            compacted = self._microcompact(messages)
            if compacted is not None:
                return {"messages": compacted}

        if strategy == MitigationStrategy.STOP:
            logger.warning("Context pressure critical — recommending stop")

        return None

    @staticmethod
    def _microcompact(messages: list[Any]) -> list[Any] | None:
        """Build a new message list with old large tool outputs truncated.

        Returns None if no changes were made.
        """
        if len(messages) <= 10:
            return None

        compact_boundary = len(messages) - 10
        # Check copy method once on first message (all messages share the same base class)
        use_model_copy = hasattr(messages[0], "model_copy")
        changed = False
        new_messages: list[Any] = []

        for i, msg in enumerate(messages):
            if i < compact_boundary:
                content = getattr(msg, "content", "")
                msg_type = getattr(msg, "type", "unknown")
                if (
                    isinstance(content, str)
                    and len(content) > 2000
                    and msg_type in ("tool", "function")
                ):
                    truncated = (
                        content[:200]
                        + f"\n...[truncated — original was {len(content):,} chars]"
                    )
                    if use_model_copy:
                        msg = msg.model_copy(update={"content": truncated})
                    else:
                        msg = msg.copy(update={"content": truncated})
                    changed = True
            new_messages.append(msg)

        return new_messages if changed else None
