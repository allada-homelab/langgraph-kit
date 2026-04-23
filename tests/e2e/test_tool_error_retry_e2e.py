"""Cluster G — ``ToolErrorMiddleware`` retries retryable failures.

The middleware retries up to ``max_retries`` times when the exception
type is in ``_RETRYABLE_TYPES`` (TimeoutError, ConnectionError, etc.).
Non-retryable exceptions fall through to the structured error
ToolMessage on the first failure.

Unit tests at ``test_resilience`` exercise the middleware with
synthetic handlers. This e2e runs a real compiled graph and asserts:

- A tool that raises ``TimeoutError`` on first call and succeeds on
  second is retried transparently (the LLM sees the success, never
  the error).
- A tool that raises a permanent error (``ValueError``) lands the
  error ToolMessage immediately (no retry).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    ToolMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, scripted_llm, tool_call_turn

pytestmark = pytest.mark.e2e


_TRANSIENT_CALL_COUNT = {"n": 0}


async def flaky_net() -> str:
    """Fail transiently on the first call, succeed on the second.

    ``TimeoutError`` is in ``_RETRYABLE_TYPES``, so
    ``ToolErrorMiddleware`` should retry once (its default
    ``max_retries=1``) and the LLM should observe the second-attempt
    success rather than a failure.
    """
    _TRANSIENT_CALL_COUNT["n"] += 1
    if _TRANSIENT_CALL_COUNT["n"] == 1:
        msg = "upstream request timed out"
        raise TimeoutError(msg)
    return "FETCHED-ON-RETRY"


async def bad_arg() -> str:
    """Raise a non-retryable ``ValueError`` every time.

    ``ValueError`` is not in ``_RETRYABLE_TYPES`` so the middleware
    must NOT retry; the error should surface on the first attempt.
    """
    msg = "bad argument permanently"
    raise ValueError(msg)


_BAD_ARG_CALLS = {"n": 0}


async def bad_arg_counted() -> str:
    """Same as bad_arg but increments a counter so we can assert no retries."""
    _BAD_ARG_CALLS["n"] += 1
    msg = "bad argument permanently"
    raise ValueError(msg)


@pytest.mark.asyncio
async def test_retryable_exception_is_retried_and_succeeds(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A TimeoutError on attempt 1 → retry → success on attempt 2.

    The LLM only ever sees the success ToolMessage. No error appears
    in state because the middleware's structured-error ToolMessage is
    built only after all attempts have been exhausted.
    """
    _TRANSIENT_CALL_COUNT["n"] = 0  # isolate between tests

    def _configure(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="flaky_net",
                name="flaky_net",
                description="Flaky tool — times out once then succeeds.",
                fn=flaky_net,
                risk=ToolRisk.READ_ONLY,
            )
        )

    scripted = scripted_llm(
        [
            tool_call_turn("flaky_net"),
            answer("got it"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="retry-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="fetch something")]},
        config={"configurable": {"thread_id": "retry-e2e"}},  # pyright: ignore[reportArgumentType]
    )

    # The tool body ran twice — once failed, once succeeded.
    assert _TRANSIENT_CALL_COUNT["n"] == 2, (
        f"Expected retryable failure to be retried exactly once;"
        f" tool body ran {_TRANSIENT_CALL_COUNT['n']} times"
    )

    # Exactly one ToolMessage for flaky_net, carrying the successful
    # payload and status != 'error'.
    flaky_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "flaky_net"
    ]
    assert len(flaky_msgs) == 1, (
        f"After retry+success the stream should carry exactly one ToolMessage;"
        f" got {len(flaky_msgs)}"
    )
    final = flaky_msgs[0]
    assert "FETCHED-ON-RETRY" in str(final.content), (
        f"Final ToolMessage should carry the retry-success payload;"
        f" got {final.content!r}"
    )
    assert getattr(final, "status", None) != "error", (
        f"Retry success must not be tagged status='error'; got {final.status!r}"
    )


@pytest.mark.asyncio
async def test_non_retryable_exception_is_not_retried(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A ValueError is NOT retried — error surfaces immediately.

    Pinning this prevents a future widening of ``_RETRYABLE_TYPES``
    from silently retrying every tool exception, which would hide
    real bugs behind duplicate network calls / side effects.
    """
    _BAD_ARG_CALLS["n"] = 0

    def _configure(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="bad_arg_counted",
                name="bad_arg_counted",
                description="Raises ValueError on every call (non-retryable).",
                fn=bad_arg_counted,
                risk=ToolRisk.READ_ONLY,
            )
        )

    scripted = scripted_llm(
        [
            tool_call_turn("bad_arg_counted"),
            answer("gave up"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="no-retry-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_tools=_configure,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="try it")]},
        config={"configurable": {"thread_id": "no-retry-e2e"}},  # pyright: ignore[reportArgumentType]
    )

    assert _BAD_ARG_CALLS["n"] == 1, (
        f"Non-retryable exceptions must not retry; tool body ran"
        f" {_BAD_ARG_CALLS['n']} times"
    )

    errored = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
    ]
    assert errored, (
        "ToolErrorMiddleware should have emitted a structured error"
        " ToolMessage for the non-retryable failure"
    )
    final_error = errored[-1]
    content = str(final_error.content)
    assert "ValueError" in content, f"Error should carry exception type; got {content!r}"
    assert "retryable: False" in content, (
        f"Error ToolMessage should tag the failure as non-retryable;"
        f" got {content!r}"
    )
