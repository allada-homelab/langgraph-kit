"""Coverage — issue #6 wiring up of orphaned ToolCapability fields.

Three previously-advisory fields are now enforced by bundled middleware:

- ``max_output_chars`` — per-tool override of the persistence threshold
- ``offload_large_results`` — opt-out of persistence regardless of size
- ``interrupt_before`` — auto-HITL pause before the tool runs

Plus a wiring smoke test for ``coordinator=True`` on
``build_deep_agent`` (the helper used to be unreachable from the
public builder).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from langgraph_kit.core.context_management.result_persistence import (
    ResultPersistenceMiddleware,
)
from langgraph_kit.core.hitl import AutoInterruptMiddleware
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Test fixtures: minimal stand-ins for the runtime objects the middlewares need
# ---------------------------------------------------------------------------


class _Runtime:
    def __init__(
        self, store: Any | None = None, thread_id: str = "test-thread"
    ) -> None:
        super().__init__()
        self.store = store
        self.config = {"configurable": {"thread_id": thread_id}}


class _Request:
    def __init__(
        self, tool_name: str, *, runtime: _Runtime, args: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self.tool_call: dict[str, Any] = {
            "id": "tc-1",
            "name": tool_name,
            "args": args or {},
        }
        self.runtime = runtime


class _SpyStore:
    """In-memory async-style store; records aput calls so tests can assert on them."""

    def __init__(self) -> None:
        super().__init__()
        self.puts: list[tuple[Any, str, dict[str, Any]]] = []

    async def aput(
        self, namespace: tuple[str, ...], key: str, value: dict[str, Any]
    ) -> None:
        self.puts.append((namespace, key, value))

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        # Used by middleware rollback path — irrelevant for these tests.
        _ = namespace, key


def _make_tool_message(content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc-1", name="dummy")


def _registry_with(*caps: ToolCapability) -> ToolRegistry:
    reg = ToolRegistry()
    for cap in caps:
        reg.register(cap)
    return reg


# ---------------------------------------------------------------------------
# (a) max_output_chars — per-tool threshold override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_tool_max_output_chars_truncates_earlier_than_default() -> None:
    """A tool with `max_output_chars=200` persists a 300-char result while
    a tool without the override stays inline (because 300 < default 4000)."""
    cap = ToolCapability(
        id="chatty.read_file",
        name="read_file",
        description="reads a file",
        fn=lambda: None,
        max_output_chars=200,
    )
    store = _SpyStore()
    mw = ResultPersistenceMiddleware(tool_registry=_registry_with(cap))

    big = "x" * 300
    runtime = _Runtime(store=store)
    request = _Request("read_file", runtime=runtime)

    async def handler(_req: Any) -> ToolMessage:
        return _make_tool_message(big)

    result = await mw.awrap_tool_call(request, handler)
    # The result should have been persisted (one aput call).
    assert len(store.puts) == 1, f"expected one persistence write, got {store.puts!r}"
    # The replacement content carries a [Full result persisted ... ref:] note.
    assert "Full result persisted" in str(result.content)


@pytest.mark.asyncio
async def test_default_threshold_used_when_no_per_tool_override() -> None:
    """Tool without ``max_output_chars`` falls back to the middleware default;
    a 300-char result is well under that and should stay inline."""
    cap = ToolCapability(
        id="quiet.echo",
        name="echo",
        description="echoes",
        fn=lambda: None,
    )
    store = _SpyStore()
    mw = ResultPersistenceMiddleware(tool_registry=_registry_with(cap))

    runtime = _Runtime(store=store)
    request = _Request("echo", runtime=runtime)

    async def handler(_req: Any) -> ToolMessage:
        return _make_tool_message("x" * 300)

    result = await mw.awrap_tool_call(request, handler)
    assert store.puts == []
    assert result.content == "x" * 300


@pytest.mark.asyncio
async def test_no_registry_falls_back_to_default_threshold() -> None:
    """When no registry is wired, behavior matches pre-change default."""
    store = _SpyStore()
    mw = ResultPersistenceMiddleware()

    runtime = _Runtime(store=store)
    request = _Request("echo", runtime=runtime)

    async def handler(_req: Any) -> ToolMessage:
        # Just over the 4000-char default → should persist.
        return _make_tool_message("x" * 4001)

    result = await mw.awrap_tool_call(request, handler)
    assert len(store.puts) == 1
    assert "Full result persisted" in str(result.content)


# ---------------------------------------------------------------------------
# (b) offload_large_results — opt-out of persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offload_false_keeps_result_inline_regardless_of_size() -> None:
    """A tool with ``offload_large_results=False`` should never persist."""
    cap = ToolCapability(
        id="opt_out.tool",
        name="tool_a",
        description="opted out",
        fn=lambda: None,
        offload_large_results=False,
    )
    store = _SpyStore()
    mw = ResultPersistenceMiddleware(tool_registry=_registry_with(cap))

    big = "x" * 10_000  # well over default 4000 threshold
    runtime = _Runtime(store=store)
    request = _Request("tool_a", runtime=runtime)

    async def handler(_req: Any) -> ToolMessage:
        return _make_tool_message(big)

    result = await mw.awrap_tool_call(request, handler)
    assert store.puts == []
    assert result.content == big


@pytest.mark.asyncio
async def test_offload_true_persists_when_over_threshold() -> None:
    cap = ToolCapability(
        id="opted_in.tool",
        name="tool_b",
        description="opted in",
        fn=lambda: None,
        offload_large_results=True,
    )
    store = _SpyStore()
    mw = ResultPersistenceMiddleware(tool_registry=_registry_with(cap))

    runtime = _Runtime(store=store)
    request = _Request("tool_b", runtime=runtime)

    async def handler(_req: Any) -> ToolMessage:
        return _make_tool_message("x" * 5000)

    result = await mw.awrap_tool_call(request, handler)
    assert len(store.puts) == 1
    assert "Full result persisted" in str(result.content)


def test_default_offload_large_results_is_true() -> None:
    """Default flipped from False to True in #6 to honor the field for
    persisted-by-default tools. Documented as a Changed entry."""
    cap = ToolCapability(
        id="default.tool",
        name="default_tool",
        description="defaults",
        fn=lambda: None,
    )
    assert cap.offload_large_results is True


# ---------------------------------------------------------------------------
# (c) AutoInterruptMiddleware — capability-driven HITL gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_interrupt_passes_through_when_flag_unset() -> None:
    cap = ToolCapability(
        id="safe.tool",
        name="safe_tool",
        description="harmless",
        fn=lambda: None,
        interrupt_before=False,
    )
    mw = AutoInterruptMiddleware(tool_registry=_registry_with(cap))

    handler_called = False

    async def handler(_req: Any) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _make_tool_message("ok")

    runtime = _Runtime()
    request = _Request("safe_tool", runtime=runtime)
    result = await mw.awrap_tool_call(request, handler)
    assert handler_called is True
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_auto_interrupt_passes_through_when_no_registry() -> None:
    """No registry means we can't know the risk profile; fail open."""
    mw = AutoInterruptMiddleware(tool_registry=None)

    handler_called = False

    async def handler(_req: Any) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _make_tool_message("ok")

    runtime = _Runtime()
    request = _Request("any_tool", runtime=runtime)
    await mw.awrap_tool_call(request, handler)
    assert handler_called is True


@pytest.mark.asyncio
async def test_auto_interrupt_pauses_and_proceeds_on_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``interrupt_before=True`` and the user accepts, the tool runs."""
    cap = ToolCapability(
        id="risky.delete",
        name="delete_thing",
        description="deletes a thing",
        fn=lambda: None,
        risk=ToolRisk.DESTRUCTIVE,
        interrupt_before=True,
    )
    mw = AutoInterruptMiddleware(tool_registry=_registry_with(cap))

    captured: list[Any] = []

    def fake_interrupt(payload: Any) -> Any:
        captured.append(payload)
        return {"type": "accept"}

    import langgraph.types as lg_types

    monkeypatch.setattr(lg_types, "interrupt", fake_interrupt)

    handler_called = False

    async def handler(_req: Any) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _make_tool_message("done")

    runtime = _Runtime()
    request = _Request("delete_thing", runtime=runtime, args={"path": "/some/path"})
    result = await mw.awrap_tool_call(request, handler)

    assert len(captured) == 1
    payload = captured[0]
    assert payload["action_request"]["action"] == "delete_thing"
    assert payload["action_request"]["args"] == {"path": "/some/path"}
    assert handler_called is True
    assert result.content == "done"


@pytest.mark.asyncio
async def test_auto_interrupt_rejects_with_tool_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User rejection returns a ToolMessage explaining the refusal so the
    agent can adjust rather than crash."""
    cap = ToolCapability(
        id="risky.rm",
        name="rm",
        description="removes",
        fn=lambda: None,
        interrupt_before=True,
    )
    mw = AutoInterruptMiddleware(tool_registry=_registry_with(cap))

    def fake_interrupt(_payload: Any) -> Any:
        return {"type": "response", "args": "no — wrong path"}

    import langgraph.types as lg_types

    monkeypatch.setattr(lg_types, "interrupt", fake_interrupt)

    handler_called = False

    async def handler(_req: Any) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _make_tool_message("oops")

    runtime = _Runtime()
    request = _Request("rm", runtime=runtime)
    result = await mw.awrap_tool_call(request, handler)

    assert handler_called is False
    assert isinstance(result, ToolMessage)
    assert "rejected" in str(result.content).lower()
    assert "wrong path" in str(result.content)


@pytest.mark.asyncio
async def test_auto_interrupt_ignore_path_returns_tool_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = ToolCapability(
        id="risky.git_push",
        name="git_push",
        description="pushes",
        fn=lambda: None,
        interrupt_before=True,
    )
    mw = AutoInterruptMiddleware(tool_registry=_registry_with(cap))

    def fake_interrupt(_payload: Any) -> Any:
        return {"type": "ignore"}

    import langgraph.types as lg_types

    monkeypatch.setattr(lg_types, "interrupt", fake_interrupt)

    handler_called = False

    async def handler(_req: Any) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _make_tool_message("pushed")

    runtime = _Runtime()
    request = _Request("git_push", runtime=runtime)
    result = await mw.awrap_tool_call(request, handler)
    assert handler_called is False
    assert "skip" in str(result.content).lower()


# ---------------------------------------------------------------------------
# (e) coordinator=True wires CoordinatorMode into build_deep_agent
# ---------------------------------------------------------------------------


def test_coordinator_get_coordinator_tools_filters_by_risk() -> None:
    """The narrowing helper used by ``coordinator=True`` returns only
    ``risk=READ_ONLY`` tools (mutating + destructive are filtered)."""
    from langgraph_kit.core.coordinator import CoordinatorMode

    def _safe_fn() -> str:
        return "safe"

    def _danger_fn() -> str:
        return "danger"

    safe = ToolCapability(
        id="r.search",
        name="search",
        description="",
        fn=_safe_fn,
        risk=ToolRisk.READ_ONLY,
    )
    danger = ToolCapability(
        id="r.delete",
        name="delete",
        description="",
        fn=_danger_fn,
        risk=ToolRisk.DESTRUCTIVE,
    )
    reg = _registry_with(safe, danger)
    coord = CoordinatorMode(reg)
    tools = coord.get_coordinator_tools()
    # ``compile_tools`` returns the underlying ``cap.fn`` callables. The
    # READ_ONLY filter must keep ``_safe_fn`` and drop ``_danger_fn``.
    assert _safe_fn in tools
    assert _danger_fn not in tools


def test_coordinator_conditions_match_section_definitions() -> None:
    from langgraph_kit.core.coordinator import COORDINATOR_SECTIONS, CoordinatorMode

    conditions = CoordinatorMode.get_conditions()
    section_conditions = {s.condition for s in COORDINATOR_SECTIONS if s.condition}
    # Every coordinator section must be covered by the activation set.
    assert section_conditions <= conditions
