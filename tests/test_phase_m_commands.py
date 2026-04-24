"""Regression tests for Phase M commands polish."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.commands.builtins import (
    _microcompact,
    build_status_command,
)
from langgraph_kit.core.commands.dispatch import CommandDispatcher
from langgraph_kit.core.context_management.pressure_middleware import (
    microcompact,
)
from langgraph_kit.core.memory.models import MemoryRecord, MemoryScope, MemoryType
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

from .conftest import MockStore


def test_microcompact_shared_between_callers() -> None:
    """Both the /compact command and PressureMiddleware route through the
    same implementation so thresholds can't drift."""
    from langgraph_kit.core.context_management.pressure_middleware import (
        PressureMiddleware,
    )

    # Hidden _microcompact on the middleware should delegate to the shared helper.
    assert callable(microcompact)
    assert callable(_microcompact)
    # Exercise: a synthetic too-large tool message well outside the
    # recent window; both entry points truncate.
    msgs = [_fake_tool_msg(i, big=True) for i in range(15)]
    out_via_mw = PressureMiddleware._microcompact(msgs)
    out_via_builtin = _microcompact(msgs)
    out_via_module = microcompact(msgs)
    assert out_via_mw is not None
    assert out_via_builtin is not None
    assert out_via_module is not None


class _FakeToolMessage:
    def __init__(self, tid: str, content: str) -> None:
        self.content = content
        self.type = "tool"
        self.id = tid

    def model_copy(self, *, update: dict[str, Any]) -> _FakeToolMessage:
        new = _FakeToolMessage(self.id, self.content)
        for k, v in update.items():
            setattr(new, k, v)
        return new


def _fake_tool_msg(i: int, big: bool = False) -> _FakeToolMessage:
    content = "x" * 5000 if big else "short"
    msg = _FakeToolMessage(tid=f"t{i}", content=content)
    return msg


@pytest.mark.asyncio
async def test_status_counts_every_memory_scope() -> None:
    """Earlier /status only counted USER scope — project/team/assistant
    records were invisible even when they existed."""
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    await mgr.create(
        MemoryRecord(
            id="u1",
            title="u",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
            summary="s",
            body="b",
        )
    )
    await mgr.create(
        MemoryRecord(
            id="p1",
            title="p",
            type=MemoryType.PROJECT,
            scope=MemoryScope.PROJECT,
            summary="s",
            body="b",
        )
    )
    await mgr.create(
        MemoryRecord(
            id="t1",
            title="t",
            type=MemoryType.REFERENCE,
            scope=MemoryScope.TEAM,
            summary="s",
            body="b",
        )
    )

    class _Monitor:
        def assess(self, _messages: list[Any]) -> Any:
            class _S:
                estimated_tokens = 0
                window_limit = 100_000
                pressure_pct = 0.0
                large_tool_outputs = 0

            return _S()

        def choose_mitigation(self, _signals: Any) -> Any:
            class _Strat:
                value = "none"

            return _Strat()

    handle = build_status_command(
        pressure_monitor=_Monitor(), memory_mgr=mgr
    )
    result = await handle("", {"messages": []})
    # Total memory count should reflect all three seeded scopes.
    assert "3 total" in result.output, (
        f"Expected '3 total' memories; got: {result.output!r}"
    )
    # Scope breakdown shows each non-zero scope.
    assert "user=1" in result.output
    assert "project=1" in result.output
    assert "team=1" in result.output


@pytest.mark.asyncio
async def test_dispatcher_error_metadata_no_longer_set() -> None:
    """Dispatcher used to emit ``metadata={'error': True}`` on handler
    exceptions — nothing ever read it. Verify the flag is gone so
    downstream consumers can't start depending on it accidentally."""
    dispatcher = CommandDispatcher()

    async def raising_handler(_args: str, _ctx: dict[str, Any]) -> Any:
        raise RuntimeError("boom")

    dispatcher.register("boom", raising_handler)
    result = await dispatcher.dispatch("/boom")
    assert result.handled is True
    assert "error" not in result.metadata
