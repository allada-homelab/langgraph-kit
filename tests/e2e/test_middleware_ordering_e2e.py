"""End-to-end scenarios for middleware stack ordering invariants.

The middleware stack is where composition-level bugs hide: each piece
can be individually correct while the combination misbehaves. These
tests exercise the ordering guarantees that downstream code relies on.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    ToolMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


class _CapturingStopHook:
    """Stop hook that records the state it was called with.

    Stored as an instance attribute so the test can inspect it after
    ``ainvoke`` returns. Non-blocking by default (``blocking`` attribute
    absent/falsy) so the middleware logs-and-continues on errors — we
    don't want a test assertion inside the hook to silently mask a
    real problem, so the test asserts AFTER the run, not inside the hook.
    """

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def on_turn_complete(self, state: Any) -> None:
        # Snapshot the state dict — LangGraph mutates/replaces entries
        # asynchronously, so we grab a shallow copy of the messages
        # list at exactly this point.
        if isinstance(state, dict):
            msgs = state.get("messages")
            self.calls.append(list(msgs) if msgs is not None else None)
        else:
            self.calls.append(state)


@pytest.mark.asyncio
async def test_save_memory_tool_persists_before_stop_hook_runs(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Stop hooks must see final state including tool writes to the Store.

    Invariant: when the LLM calls ``save_memory(...)``, the tool runs
    synchronously (memory manager persists the record), the
    corresponding ``ToolMessage`` lands in state, THEN
    ``StopHooksMiddleware.aafter_agent`` fires and the hook sees the
    post-write state. Every piece of downstream automation that relies
    on "read the store after a turn completes" depends on this
    ordering.

    What would break this: a middleware reordering that puts stop hooks
    earlier than the tool-execution phase, or a future refactor that
    turns save_memory into a deferred/async persist without updating
    the hook contract.
    """
    hook = _CapturingStopHook()

    scripted = scripted_llm(
        [
            tool_call_turn(
                "save_memory",
                {
                    "title": "pi",
                    "memory_type": "reference",
                    "scope": "user",
                    "summary": "approximate value of pi",
                    "body": "pi=3.14159",
                },
            ),
            answer("saved"),
        ]
    )

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="ordering-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            stop_hooks=[hook],
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="remember pi=3.14159")]},
        config={"configurable": {"thread_id": "ordering-1"}},  # pyright: ignore[reportArgumentType]
    )

    # 1. Tool actually ran — its ToolMessage is in the final state.
    tool_msg = assert_tool_invoked(result, "save_memory")
    assert "pi" in str(tool_msg.content).lower() or "sav" in str(tool_msg.content).lower(), (
        f"save_memory ToolMessage content doesn't look like a save confirmation: "
        f"{tool_msg.content!r}"
    )

    # 2. Memory landed in the store (MockStore holds whatever the memory
    # manager put in). The exact namespace is implementation-detail, so
    # scan all namespaces for a record mentioning our content.
    found = _find_memory(e2e_store, "3.14159")
    assert found, (
        "save_memory tool ran but nothing matching the content reached the "
        f"MockStore. Namespaces observed: {list(e2e_store._data.keys())}"
    )

    # 3. Stop hook fired at least once, and at the time it saw state, the
    # save_memory ToolMessage was already present. This is the ordering
    # invariant: hook sees post-tool state, not pre-tool.
    assert hook.calls, "StopHooksMiddleware never called on_turn_complete"
    seen_tool_msgs = [
        m
        for snapshot in hook.calls
        if snapshot is not None
        for m in snapshot
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "save_memory"
    ]
    assert seen_tool_msgs, (
        "Stop hook fired but never saw the save_memory ToolMessage in any "
        "state snapshot — the middleware ordering invariant is broken. "
        f"Hook calls: {len(hook.calls)}"
    )


def _find_memory(store: Any, marker: str) -> bool:
    """Scan a MockStore for any value whose content matches ``marker``."""
    for namespace, entries in store._data.items():
        for key, value in entries.items():
            # Values are dicts (MemoryRecord dumps). Flatten via str().
            if marker in str(value):
                return True
            _ = key
            _ = namespace
    return False
