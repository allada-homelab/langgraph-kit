"""Stop hooks lifecycle middleware.

Runs registered lifecycle hooks at turn boundaries.
Non-blocking hooks log failures; blocking hooks propagate exceptions.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware

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
