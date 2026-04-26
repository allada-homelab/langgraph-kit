"""Unit tests for ``SystemContextProvider``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider


@pytest.mark.asyncio
async def test_provide_includes_expected_fields() -> None:
    """Output contains datetime, platform, OS name, and a kit version line."""
    out = await SystemContextProvider().provide({})

    assert out.startswith("# System Context")
    assert "Current time (UTC):" in out
    assert "Platform:" in out
    assert "OS name:" in out
    assert "Kit version:" in out


@pytest.mark.asyncio
async def test_provide_is_deterministic_with_fixed_clock() -> None:
    """Two calls with a frozen clock return identical output.

    The agent prompt is composed multiple times within a turn (e.g.
    coordinator re-render, sub-worker pass). If the system context
    drifts between those passes, prompt-cache reuse breaks. Pinning a
    clock here guards that contract.
    """
    fixed = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
    provider = SystemContextProvider(clock=lambda: fixed)

    first = await provider.provide({})
    second = await provider.provide({})

    assert first == second
    assert "2026-04-26T12:00:00+00:00" in first


@pytest.mark.asyncio
async def test_provide_advances_with_real_clock() -> None:
    """Default clock returns *now*; output is non-empty."""
    out = await SystemContextProvider().provide({})
    assert "Current time (UTC):" in out
