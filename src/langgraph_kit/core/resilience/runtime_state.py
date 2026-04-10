"""Runtime state tracking middleware.

Tracks structured state transitions and emits runtime metadata.
Uses contextvars for per-request state so concurrent invocations don't interfere.
"""

from __future__ import annotations

import contextvars
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware


class RuntimeStateMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Tracks structured state transitions and emits runtime metadata.

    Uses contextvars for per-request state so concurrent invocations don't interfere.
    """

    _cv_state: contextvars.ContextVar[str] = contextvars.ContextVar(
        "runtime_state", default="idle"
    )
    _cv_stop_reason: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "runtime_stop_reason", default=None
    )
    _cv_turn_count: contextvars.ContextVar[int] = contextvars.ContextVar(
        "runtime_turn_count", default=0
    )

    @property
    def state(self) -> str:
        return self._cv_state.get()

    @property
    def stop_reason(self) -> str | None:
        return self._cv_stop_reason.get()

    @property
    def turn_count(self) -> int:
        return self._cv_turn_count.get()

    async def abefore_agent(self, _state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        self._cv_state.set("started")
        self._cv_stop_reason.set(None)
        self._cv_turn_count.set(self._cv_turn_count.get() + 1)
        return None

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self._cv_state.set("streaming")
        try:
            result = await handler(request)
            self._cv_state.set("completed")
            self._cv_stop_reason.set("final_answer")
            return result
        except Exception as exc:
            self._cv_state.set("failed")
            self._cv_stop_reason.set(f"error: {type(exc).__name__}")
            raise

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        prev = self._cv_state.get()
        self._cv_state.set("tool_running")
        try:
            result = await handler(request)
            self._cv_state.set(prev)
            return result
        except Exception:
            self._cv_state.set("tool_failed")
            raise
