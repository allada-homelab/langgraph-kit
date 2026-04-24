"""Regression tests for Phase J tools fixes."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.deferred import (
    DeferredToolRegistry,
    build_call_deferred_tool,
)
from langgraph_kit.core.tools.registry import ToolRegistry


def _plain_cap(
    *, cap_id: str, risk: ToolRisk = ToolRisk.READ_ONLY, tags: list[str] | None = None
) -> ToolCapability:
    async def fn(**_: Any) -> str:
        return "ok"

    return ToolCapability(
        id=cap_id,
        name=cap_id,
        description=cap_id,
        fn=fn,
        risk=risk,
        tags=tags or [],
    )


def test_compile_tools_applies_tags_filter() -> None:
    reg = ToolRegistry()
    reg.register(_plain_cap(cap_id="git", tags=["git"]))
    reg.register(_plain_cap(cap_id="mcp", tags=["mcp"]))

    # Without filter: 2 tools.
    assert len(reg.compile_tools()) == 2
    # With tag filter: only git tool returned.
    filtered = reg.compile_tools(tags={"git"})
    assert len(filtered) == 1


def test_collect_prompt_fragments_applies_tags_filter() -> None:
    reg = ToolRegistry()
    cap_a = _plain_cap(cap_id="git-a", tags=["git"])
    cap_b = _plain_cap(cap_id="mcp-b", tags=["mcp"])
    # Give them guidance so fragments are emitted.
    cap_a = cap_a.model_copy(update={"prompt_guidance": "use for git"})
    cap_b = cap_b.model_copy(update={"prompt_guidance": "use for mcp"})
    reg.register(cap_a)
    reg.register(cap_b)

    out = reg.collect_prompt_fragments(tags={"git"})
    assert "use for git" in out
    assert "use for mcp" not in out


@pytest.mark.asyncio
async def test_deferred_registry_blocks_destructive_by_default() -> None:
    reg = DeferredToolRegistry()
    reg.register(_plain_cap(cap_id="wipe-db", risk=ToolRisk.DESTRUCTIVE))

    call = build_call_deferred_tool(reg)
    result = await call(tool_id="wipe-db", arguments={})

    assert "destructive" in result.lower()
    assert "allow_destructive" in result


@pytest.mark.asyncio
async def test_deferred_registry_invokes_destructive_when_opted_in() -> None:
    reg = DeferredToolRegistry(allow_destructive=True)
    reg.register(_plain_cap(cap_id="wipe-db", risk=ToolRisk.DESTRUCTIVE))

    call = build_call_deferred_tool(reg)
    result = await call(tool_id="wipe-db", arguments={})

    # No error surface — the fake fn returns "ok".
    assert "destructive" not in result.lower()
