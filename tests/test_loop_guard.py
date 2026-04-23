"""Tests for ToolLoopGuardMiddleware — soft loop detection on repeated tool calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph_kit.core.resilience.loop_guard import (
    DEFAULT_LOOP_THRESHOLD,
    ToolLoopGuardMiddleware,
)


_SHARED_RUNTIME = object()
"""Stable identity for all ``_request`` calls in this module.

The loop guard now keys its streak counter on ``id(request.runtime)``
so two concurrent real-graph invocations (each with its own Runtime
dataclass) stay isolated. Unit tests simulate a single run by sharing
one sentinel across all ``_request`` calls; tests that want to model a
boundary (e.g. ``abefore_agent_resets_streak_between_turns``) rely on
the middleware's explicit clear, not on a new runtime.
"""


@pytest.fixture(autouse=True)
def _isolate_guard_state() -> Any:
    """Wipe the module-level streak store between tests.

    The guard keys counters on thread_id; tests using ``_SHARED_RUNTIME``
    collapse to ``id(_SHARED_RUNTIME)``. Without this fixture a prior
    test's residual entry would poison the next test's first call.
    """
    from langgraph_kit.core.resilience import loop_guard

    loop_guard._streaks.clear()
    yield
    loop_guard._streaks.clear()


def _request(tool_name: str) -> Any:
    req = MagicMock()
    req.tool_call = {"name": tool_name, "args": {}}
    req.runtime = _SHARED_RUNTIME
    return req


def _result(content: str = "ok") -> Any:
    """Minimal stand-in for a ToolMessage — has .content and model_copy."""
    msg = MagicMock()
    msg.content = content
    msg.model_copy = lambda *, update: MagicMock(content=update["content"])
    return msg


@pytest.mark.asyncio
async def test_below_threshold_result_untouched() -> None:
    """Under the threshold, the middleware is a strict pass-through.

    The original result object must come back byte-for-byte — we're not
    going to nudge the agent for 3 tool_search calls, that's normal
    exploration.
    """
    mw = ToolLoopGuardMiddleware(threshold=5)
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    original = _result("match for 'deploy'")
    handler = AsyncMock(return_value=original)

    for _ in range(4):
        result = await mw.awrap_tool_call(_request("tool_search"), handler)
        assert result is original, (
            "Below threshold the middleware must be a strict pass-through"
        )


@pytest.mark.asyncio
async def test_at_threshold_appends_advisory() -> None:
    """Hitting the threshold appends the advisory to the returned content.

    The underlying tool still runs — the advisory is a *soft* nudge.
    The model sees its real search result with a trailing paragraph
    suggesting it try a different approach.
    """
    mw = ToolLoopGuardMiddleware(threshold=5)
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    handler = AsyncMock(side_effect=lambda _req: _result("search hit"))

    # First 4 are pass-through
    for _ in range(4):
        await mw.awrap_tool_call(_request("tool_search"), handler)

    # 5th triggers the advisory
    result = await mw.awrap_tool_call(_request("tool_search"), handler)
    assert "search hit" in result.content, (
        "Original tool output must be preserved — guard is non-breaking"
    )
    assert "tool_search" in result.content
    assert "5 times in a row" in result.content
    # Call count: handler must have been called 5 times — the guard
    # does NOT skip the underlying tool call.
    assert handler.await_count == 5


@pytest.mark.asyncio
async def test_different_tool_resets_streak() -> None:
    """A call to any OTHER tool breaks the streak.

    The agent made forward progress on something else, so the counter
    goes back to zero and we don't hassle it on the next tool_search.
    """
    mw = ToolLoopGuardMiddleware(threshold=3)
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    handler = AsyncMock(side_effect=lambda _req: _result("ok"))

    await mw.awrap_tool_call(_request("tool_search"), handler)
    await mw.awrap_tool_call(_request("tool_search"), handler)
    # Different tool — streak resets here
    await mw.awrap_tool_call(_request("other_tool"), handler)
    # Back to tool_search — this is call 1 of the NEW streak, not 3 of 3
    result = await mw.awrap_tool_call(_request("tool_search"), handler)
    assert "times in a row" not in result.content


@pytest.mark.asyncio
async def test_abefore_agent_resets_streak_between_turns() -> None:
    """A new turn zeros the counter — the streak is per-turn, not per-session.

    Without this, a user who happened to trigger 4 tool_searches on one
    turn would see the advisory injected on their very first
    tool_search of the next turn. Each user turn is a fresh context.
    """
    mw = ToolLoopGuardMiddleware(threshold=3)
    handler = AsyncMock(side_effect=lambda _req: _result("ok"))

    # Turn 1
    await mw.abefore_agent({}, _SHARED_RUNTIME)
    for _ in range(2):
        await mw.awrap_tool_call(_request("tool_search"), handler)

    # Turn 2 — counter reset
    await mw.abefore_agent({}, _SHARED_RUNTIME)
    result = await mw.awrap_tool_call(_request("tool_search"), handler)
    assert "times in a row" not in result.content


@pytest.mark.asyncio
async def test_threshold_zero_disables_guard() -> None:
    """``threshold=0`` is the kill switch — the middleware is transparent."""
    mw = ToolLoopGuardMiddleware(threshold=0)
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    original = _result("r")
    handler = AsyncMock(return_value=original)

    for _ in range(50):
        result = await mw.awrap_tool_call(_request("tool_search"), handler)
        assert result is original


@pytest.mark.asyncio
async def test_custom_tool_name_and_threshold() -> None:
    """Guard can be reused on any tool name with any threshold."""
    mw = ToolLoopGuardMiddleware(
        tool_name="list_memories", threshold=2, advice="LOOP on {tool_name} x{count}"
    )
    await mw.abefore_agent({}, _SHARED_RUNTIME)
    handler = AsyncMock(side_effect=lambda _req: _result("ok"))

    # tool_search doesn't trigger this guard (different tool_name)
    await mw.awrap_tool_call(_request("tool_search"), handler)
    # First list_memories — under threshold
    result = await mw.awrap_tool_call(_request("list_memories"), handler)
    assert "LOOP" not in result.content
    # Second list_memories — threshold hit
    result = await mw.awrap_tool_call(_request("list_memories"), handler)
    assert result.content.endswith("LOOP on list_memories x2")


@pytest.mark.asyncio
async def test_advisory_goes_to_string_tool_outputs() -> None:
    """Some tool wrappers return a bare string instead of a ToolMessage.

    The middleware must append to strings too — many of the kit's own
    tools (like ``create_artifact``, ``save_memory``) return plain str.
    """
    mw = ToolLoopGuardMiddleware(threshold=1, advice="LOOP")
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    handler = AsyncMock(return_value="raw string output")
    result = await mw.awrap_tool_call(_request("tool_search"), handler)
    assert isinstance(result, str)
    assert result.startswith("raw string output")
    assert "LOOP" in result


@pytest.mark.asyncio
async def test_default_threshold_is_five() -> None:
    """Documented default must actually be 5 — this is part of the API contract."""
    assert DEFAULT_LOOP_THRESHOLD == 5
    mw = ToolLoopGuardMiddleware()
    # We don't inspect the private attr, we verify the behavior: 4 pass
    # through, 5 triggers.
    await mw.abefore_agent({}, _SHARED_RUNTIME)
    handler = AsyncMock(side_effect=lambda _req: _result("x"))
    for _ in range(4):
        result = await mw.awrap_tool_call(_request("tool_search"), handler)
        assert "times in a row" not in result.content
    result = await mw.awrap_tool_call(_request("tool_search"), handler)
    assert "5 times in a row" in result.content


@pytest.mark.asyncio
async def test_builder_stack_wires_the_guard() -> None:
    """End-to-end: ``build_middleware_stack`` must include the guard.

    Without this the middleware would exist but never run — the same
    class of bug the extension audit surfaced elsewhere. Callers pass
    ``tool_search_loop_threshold`` through and the guard respects it.
    """
    from langgraph_kit.core.context_management.pressure import PressureMonitor
    from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
    from langgraph_kit.core.memory.persistent import PersistentMemoryManager

    middleware, _ = build_middleware_stack(
        llm=MagicMock(),
        memory_mgr=PersistentMemoryManager(MagicMock()),
        pressure_monitor=PressureMonitor(),
        tool_search_loop_threshold=3,
    )
    guards = [m for m in middleware if isinstance(m, ToolLoopGuardMiddleware)]
    assert len(guards) == 1
    # The configured threshold is applied, not the default.
    # (We exercise it behaviorally rather than asserting a private attribute.)
    guard = guards[0]
    await guard.abefore_agent({}, _SHARED_RUNTIME)
    handler = AsyncMock(side_effect=lambda _req: _result("x"))
    for _ in range(2):
        result = await guard.awrap_tool_call(_request("tool_search"), handler)
        assert "times in a row" not in result.content
    result = await guard.awrap_tool_call(_request("tool_search"), handler)
    assert "3 times in a row" in result.content


@pytest.mark.asyncio
async def test_streak_persists_across_sibling_asyncio_tasks() -> None:
    """Guard must count across tool calls run as sibling asyncio tasks.

    LangGraph schedules each tool call as its own asyncio task. The
    previous implementation called ``ContextVar.set(new_dict)`` inside
    ``awrap_tool_call`` — and ``set`` is copy-on-write per context, so
    each child task created its own fresh dict and saw ``count=1``
    forever. The guard never fired under real graph execution; unit
    tests missed this because they ran every call in one coroutine.

    The current implementation seeds a mutable dict in
    ``abefore_agent`` (parent task) and **mutates it in place** from
    each tool-call task. Child tasks inherit a context snapshot, but
    the *values* in that snapshot (the dict reference) are shared, so
    in-place mutations are visible across siblings. This test exercises
    that exact shape.
    """
    import asyncio

    mw = ToolLoopGuardMiddleware(threshold=3)
    # Seed the shared dict in the parent context — tasks spawned below
    # inherit its reference.
    await mw.abefore_agent({}, _SHARED_RUNTIME)

    handler = AsyncMock(side_effect=lambda _req: _result("x"))

    async def _one_call() -> Any:
        return await mw.awrap_tool_call(_request("tool_search"), handler)

    results = []
    for _ in range(3):
        results.append(await asyncio.create_task(_one_call()))

    assert "3 times in a row" in results[-1].content, (
        "Streak did NOT persist across sibling tasks — this is the exact "
        "shape that broke the original ContextVar.set()-based guard under "
        "real LangGraph execution. Got: " + str(results[-1].content)
    )


@pytest.mark.asyncio
async def test_streak_isolated_between_concurrent_runs() -> None:
    """Two concurrent ``ainvoke`` executions must not share streak state.

    LangGraph's ``ainvoke`` wraps each graph run in its own context
    (runtime config, state, etc.). ContextVar-based state is therefore
    naturally isolated between concurrent runs as long as each run
    starts from a fresh ``abefore_agent``. We simulate that by running
    two scenarios in independent ``contextvars.Context`` instances.
    """
    import asyncio
    import contextvars

    mw = ToolLoopGuardMiddleware(threshold=3)
    handler = AsyncMock(side_effect=lambda _req: _result("x"))

    async def run_one() -> Any:
        await mw.abefore_agent({}, _SHARED_RUNTIME)
        last = None
        for _ in range(3):
            last = await mw.awrap_tool_call(_request("tool_search"), handler)
        return last

    # Copy parent context into two independent snapshots before spawning.
    # ``Context.run(...)`` executes the coroutine in that isolated
    # context so ``_streak`` writes in one run don't leak into the other.
    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    # ``Context.run`` is synchronous; we schedule each coroutine inside
    # its own context via ``asyncio.Task`` constructor's ``context=`` arg.
    loop = asyncio.get_event_loop()
    task_a = loop.create_task(run_one(), context=ctx_a)
    task_b = loop.create_task(run_one(), context=ctx_b)
    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert "3 times in a row" in result_a.content
    assert "3 times in a row" in result_b.content
