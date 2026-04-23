"""Cluster G — stop hook error-path e2e scenarios.

``test_middleware_ordering_e2e.py`` covers the happy-path contract
(hook fires after tool execution, sees post-write state). This file
covers the error paths: non-blocking hook raises (logged and
swallowed, run continues) vs blocking hook raises (propagates).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, last_ai_message, scripted_llm

pytestmark = pytest.mark.e2e


class _NonBlockingFailingHook:
    """Hook that always raises. ``blocking`` attribute absent = non-blocking."""

    def __init__(self) -> None:
        self.calls = 0

    async def on_turn_complete(self, state: Any) -> None:  # noqa: ARG002
        self.calls += 1
        msg = "non-blocking boom"
        raise RuntimeError(msg)


class _BlockingFailingHook:
    """Hook flagged ``blocking=True`` — exceptions should propagate."""

    blocking = True

    async def on_turn_complete(self, state: Any) -> None:  # noqa: ARG002
        msg = "blocking boom"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_non_blocking_stop_hook_failure_does_not_kill_the_run(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A non-blocking hook that raises is logged and swallowed; run completes.

    Without this behavior a buggy hook could brick an entire user
    session. The middleware's ``hasattr(hook, "blocking")`` guard
    plus ``logger.exception`` and a swallow is the contract downstream
    apps depend on.
    """
    hook = _NonBlockingFailingHook()
    scripted = scripted_llm([answer("ok")])
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="hook-nonblocking-err",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            stop_hooks=[hook],
        )

    # Should NOT raise.
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="go")]},
        config={"configurable": {"thread_id": "hook-nb"}},  # pyright: ignore[reportArgumentType]
    )
    assert "ok" in str(last_ai_message(result).content)
    assert hook.calls == 1, (
        f"Hook should fire exactly once per invoke even on failure; got {hook.calls}"
    )


@pytest.mark.asyncio
async def test_blocking_stop_hook_failure_propagates(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A ``blocking=True`` hook that raises terminates the invocation.

    Blocking hooks are used for must-succeed lifecycle work — audit
    logging, checkpoint persistence. If they fail, downstream code
    needs to know rather than silently moving on.
    """
    hook = _BlockingFailingHook()
    scripted = scripted_llm([answer("would-succeed-if-no-hook")])

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="hook-blocking-err",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            stop_hooks=[hook],
        )

    with pytest.raises(RuntimeError, match="blocking boom"):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "hook-b"}},  # pyright: ignore[reportArgumentType]
        )
