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

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware

logger = logging.getLogger(__name__)


DEFAULT_LOOP_THRESHOLD = 5


# Per-run streak counter keyed by thread_id, with tool-name as the
# inner dict key. One entry per active thread_id; cleared by
# :meth:`ToolLoopGuardMiddleware.abefore_agent` at run start so a new
# user turn never inherits the previous turn's streak, and (best-effort)
# by :meth:`~ToolLoopGuardMiddleware.aafter_agent` at run end so the
# module-level store stays bounded.
#
# Why thread_id and not something narrower:
#   - LangGraph schedules each tool call as its own asyncio task.
#     ``ContextVar.set()`` is copy-on-write per context, so a counter
#     stored via ``_var.set(new_dict)`` never propagates to sibling
#     tool-call tasks — every call saw ``count=1``. Mutating a
#     context-inherited dict works in theory but in practice each tool
#     call's task does NOT inherit the agent's abefore_agent context,
#     so the shared-dict-via-ContextVar trick also fails.
#   - ``id(request.runtime)`` varies per tool call because LangGraph
#     constructs a fresh ``ToolRuntime`` each time.
#   - ``thread_id`` is stable across ``abefore_agent`` → every
#     ``awrap_tool_call`` → ``aafter_agent`` within one ``ainvoke`` and
#     it uniquely identifies the conversation thread, giving natural
#     isolation between concurrent users.
#
# The original implementation used a ``ContextVar[dict]`` and worked
# under unit tests (all calls in one coroutine) but silently failed
# under real graph execution. Unit tests are augmented in
# ``test_loop_guard.py::test_streak_persists_across_sibling_asyncio_tasks``
# to catch regressions of this class.
_streaks: dict[Any, dict[str, int]] = {}

# Soft ceiling on the number of thread_id entries kept in ``_streaks``.
# Normally ``aafter_agent`` clears the entry at the end of every run, but
# a run that raises out without finishing (e.g. graph-level exception,
# asyncio.CancelledError) can leave its entry behind. Without a cap,
# long-lived processes under crashy traffic accumulate keys forever.
# 1000 is generous: each entry is at most a small dict of ``{tool_name: int}``
# counters, so the steady-state footprint is tiny even at the ceiling.
_STREAKS_SOFT_CAP = 1000


_MISSING_THREAD_ID = "__loop_guard_no_thread_id__"


def _thread_id_from_runtime(runtime: Any) -> Any:
    """Extract thread_id from a Runtime (abefore_agent / aafter_agent path).

    ``Runtime.execution_info.thread_id`` is set by LangGraph during graph
    execution. Falls back to a sentinel when unavailable so the guard
    still keys consistently in direct-unit-test call sites that pass
    ``None`` or a plain sentinel.
    """
    if runtime is None:
        return _MISSING_THREAD_ID
    exec_info = getattr(runtime, "execution_info", None)
    tid = getattr(exec_info, "thread_id", None) if exec_info is not None else None
    return tid if tid is not None else id(runtime)


def _thread_id_from_request(request: Any) -> Any:
    """Extract thread_id from a ToolCallRequest (awrap_tool_call path).

    ``request.runtime.config["configurable"]["thread_id"]`` is populated
    by LangGraph for every tool call in a run. Falls back to the
    runtime's identity when the config shape isn't what we expect, so
    direct-unit-test call sites keep working.
    """
    runtime = getattr(request, "runtime", None)
    if runtime is None:
        return _MISSING_THREAD_ID
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        tid = config.get("configurable", {}).get("thread_id")
        if tid is not None:
            return tid
    return id(runtime)


def _streak_bump(key: Any, name: str) -> int:
    # FIFO eviction when the dict is about to grow past the cap. Dict
    # preserves insertion order (Python 3.7+), so ``next(iter(...))``
    # returns the oldest entry — typically a crashed run's leftover.
    if key not in _streaks and len(_streaks) >= _STREAKS_SOFT_CAP:
        try:
            oldest = next(iter(_streaks))
            del _streaks[oldest]
        except StopIteration:  # pragma: no cover — _streaks non-empty by check
            pass
    counters = _streaks.setdefault(key, {})
    counters[name] = counters.get(name, 0) + 1
    return counters[name]


def _streak_reset(key: Any, name: str) -> None:
    """Zero the counter for ``name`` because a different tool ran."""
    counters = _streaks.get(key)
    if counters is None:
        return
    counters.pop(name, None)


def _streak_clear(key: Any) -> None:
    """Drop the per-run counter dict."""
    _streaks.pop(key, None)


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
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Drop any stale streak entry for this thread at run start.

        Each user turn (ainvoke) starts fresh, even on a thread that
        previously accumulated a streak. ``aafter_agent`` normally
        clears first, but we defensively clear here too so an
        incomplete prior run can't poison the new one.
        """
        _streak_clear(_thread_id_from_runtime(runtime))
        return None

    async def aafter_agent(
        self,
        state: Any,  # noqa: ARG002
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Drop this thread's counters so ``_streaks`` stays bounded."""
        _streak_clear(_thread_id_from_runtime(runtime))
        return None

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Observe each tool call, append advisory when streak exceeds threshold."""
        if self._threshold <= 0:
            return await handler(request)

        tool_call = getattr(request, "tool_call", None)
        called = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        key = _thread_id_from_request(request)

        if called != self._tool_name:
            # Any non-watched tool resets the streak — the agent made
            # forward progress on something else.
            _streak_reset(key, self._tool_name)
            return await handler(request)

        count = _streak_bump(key, self._tool_name)
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
