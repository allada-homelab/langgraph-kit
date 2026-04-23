"""Soft loop-detection middleware for repeated tool calls.

When the agent calls the same tool many times in a row with no other
activity between, it's usually stuck — exploring a dead-end search
space, re-querying after a misread result, or oscillating on the same
plan step. The :class:`ToolLoopGuardMiddleware` notices this pattern
and appends an advisory to the tool's return content after a
configurable threshold. It never cancels, raises, or removes the tool
call — the agent stays in control, it just gets a nudge to try
something else.

Default target is ``tool_search`` because it was the most common
observed loop: the agent discovers a tool via ``tool_search``, the
dispatcher call fails or the model misreads the id, and the agent
keeps searching instead of inspecting the failure. The target tool and
threshold are both configurable so callers can reuse the same
middleware for any tool that turns out to be loop-prone in practice
(e.g. ``list_memories``, ``list_async_tasks``, ``read_file``).
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware

logger = logging.getLogger(__name__)


DEFAULT_LOOP_THRESHOLD = 5


# Per-request counter keyed by tool name. ContextVar keeps concurrent
# invocations isolated — two simultaneous chat requests do NOT share
# each other's counters. Default is ``None`` (not ``{}``) so the
# contextvar doesn't share a single mutable dict across contexts —
# :func:`_streak_snapshot` materializes a fresh empty dict per reader.
_streak: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "tool_loop_streak", default=None
)


def _streak_snapshot() -> dict[str, int]:
    value = _streak.get()
    return value if value is not None else {}


def _streak_bump(name: str) -> int:
    # Copy-on-write so one request can't mutate another's view mid-flight.
    updated = {**_streak_snapshot(), name: _streak_snapshot().get(name, 0) + 1}
    _streak.set(updated)
    return updated[name]


def _streak_reset(name: str) -> None:
    """Zero the counter for ``name`` because a different tool ran."""
    current = _streak_snapshot()
    if name not in current:
        return
    _streak.set({k: v for k, v in current.items() if k != name})


class ToolLoopGuardMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Appends a soft nudge after ``threshold`` consecutive calls to a tool.

    Non-breaking by design: the tool still runs, its output still reaches
    the model, and the middleware only *appends* an advisory string so
    the model can notice the pattern on its next turn. Once the agent
    calls a different tool the streak resets naturally.

    Parameters
    ----------
    tool_name:
        Name of the tool to watch. Defaults to ``"tool_search"``.
    threshold:
        Number of consecutive calls that triggers the advisory. Defaults
        to :data:`DEFAULT_LOOP_THRESHOLD` (5). Values ``<= 0`` disable
        the guard entirely.
    advice:
        Optional override for the advisory text. Supports ``{tool_name}``
        and ``{count}`` format placeholders. If omitted, a generic
        suggestion is used.

    Examples
    --------
    >>> guard = ToolLoopGuardMiddleware(tool_name="tool_search", threshold=5)
    >>> # After the 5th consecutive tool_search the model will see its
    >>> # result with a trailing "You've called tool_search 5 times in a
    >>> # row — try an alternate approach" paragraph appended.
    """

    def __init__(
        self,
        *,
        tool_name: str = "tool_search",
        threshold: int = DEFAULT_LOOP_THRESHOLD,
        advice: str | None = None,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._threshold = threshold
        self._advice_template = advice or (
            "_Heads up:_ you've called `{tool_name}` {count} times in a row. "
            "This often indicates a loop. Before calling it again, consider:\n"
            "- If you've already discovered a relevant capability, invoke it "
            "directly (e.g. `call_deferred_tool(tool_id=..., arguments={{...}})` "
            "for deferred tools) rather than re-searching.\n"
            "- Re-read the last tool result carefully — the information you need "
            "may already be there.\n"
            "- If the current approach isn't working, try a materially different "
            "strategy or ask the user to clarify what they need."
        )

    async def abefore_agent(
        self,
        state: Any,  # noqa: ARG002
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Reset all streak counters at the start of a turn."""
        _streak.set(None)
        return None

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Observe each tool call, append advisory when streak exceeds threshold."""
        if self._threshold <= 0:
            return await handler(request)

        tool_call = getattr(request, "tool_call", None)
        called = tool_call.get("name", "") if isinstance(tool_call, dict) else ""

        if called != self._tool_name:
            # Any non-watched tool resets the streak — the agent made
            # forward progress on something else.
            _streak_reset(self._tool_name)
            return await handler(request)

        count = _streak_bump(self._tool_name)
        result = await handler(request)

        if count < self._threshold:
            return result

        advice = self._advice_template.format(tool_name=self._tool_name, count=count)
        logger.info(
            "ToolLoopGuard: %s called %d times in a row — appending advisory",
            self._tool_name,
            count,
        )
        return _append_advice(result, advice)


def _append_advice(result: Any, advice: str) -> Any:
    """Return a copy of ``result`` with ``advice`` appended to its text content.

    Works for LangChain message objects (``.content`` + ``model_copy``),
    plain strings, and anything else that cleanly supports
    concatenation. Falls back to returning the original result if no
    safe append path exists — never raises, since the whole point of
    this middleware is to be non-breaking.
    """
    if isinstance(result, str):
        return result + "\n\n" + advice

    content = getattr(result, "content", None)
    if isinstance(content, str):
        new_content = content + "\n\n" + advice
        if hasattr(result, "model_copy"):
            try:
                return result.model_copy(update={"content": new_content})
            except Exception:
                logger.debug("model_copy failed on result type %s", type(result))
        # Last-resort mutation. Works for simple mocks; preserves identity
        # so reducers don't treat it as a new message.
        try:
            result.content = new_content
        except Exception:
            logger.debug("Could not mutate .content on result type %s", type(result))
        return result

    return result
