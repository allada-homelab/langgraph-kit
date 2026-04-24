"""Regression: per-run counters in resilience middlewares must reset.

Before this fix, ``CompletionGuardMiddleware._challenges_issued`` and
``EmptyTurnMiddleware._nudge_count`` were set once in ``__init__`` and
never reset between agent runs. A middleware instance is typically
built once per compiled graph and reused across many invocations, so
after two runs that hit ``_MAX_CHALLENGES`` / ``max_nudges`` the guard
silently stopped firing for the rest of the process's lifetime.

Adding ``abefore_agent`` that zeros the counter at the start of each
run ensures the caps stay per-run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_kit.core.resilience.completion_guard import (
    CompletionGuardMiddleware,
)
from langgraph_kit.core.resilience.empty_turn import EmptyTurnMiddleware


@pytest.mark.asyncio
async def test_completion_guard_resets_challenges_between_runs() -> None:
    mw = CompletionGuardMiddleware(min_tool_calls=5)

    # Simulate exhausting the per-run budget.
    mw._challenges_issued = 2
    assert mw._challenges_issued == 2

    await mw.abefore_agent(state={}, runtime=MagicMock())

    assert mw._challenges_issued == 0, (
        "abefore_agent must reset _challenges_issued so the per-run cap is "
        "actually per-run, not lifetime-of-instance."
    )


@pytest.mark.asyncio
async def test_completion_guard_fires_on_second_run_after_first_exhausted() -> None:
    """End-to-end: after a first run hits MAX_CHALLENGES, a second run still guards."""
    mw = CompletionGuardMiddleware(min_tool_calls=5)
    runtime = MagicMock()

    # Exhaust first run manually.
    mw._challenges_issued = 2

    # Start second run.
    await mw.abefore_agent(state={}, runtime=runtime)

    # Build a state that looks premature: completion phrase + no tool calls +
    # enough messages to clear the warm-up threshold.
    messages: list[Any] = [HumanMessage(content=f"request {i}") for i in range(5)] + [
        AIMessage(content="I'm done.")
    ]
    state = {"messages": messages}

    result = await mw.aafter_model(state=state, runtime=runtime)

    assert result is not None, (
        "After reset, the guard should challenge the premature completion "
        "even though the instance previously maxed out."
    )
    assert mw._challenges_issued == 1


@pytest.mark.asyncio
async def test_empty_turn_resets_nudge_count_between_runs() -> None:
    mw = EmptyTurnMiddleware(max_nudges=2)

    mw._nudge_count = 2
    await mw.abefore_agent(state={}, runtime=MagicMock())
    assert mw._nudge_count == 0


@pytest.mark.asyncio
async def test_empty_turn_fires_on_second_run_after_first_exhausted() -> None:
    mw = EmptyTurnMiddleware(max_nudges=1)
    runtime = MagicMock()

    mw._nudge_count = 1  # Already at max
    await mw.abefore_agent(state={}, runtime=runtime)

    # Trigger empty-turn path.
    state = {"messages": [AIMessage(content="")]}
    result = await mw.aafter_model(state=state, runtime=runtime)

    assert result is not None, (
        "After reset, the second run's first empty turn should still nudge."
    )
    assert mw._nudge_count == 1
