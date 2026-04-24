"""Regression: the global ``_streaks`` dict must stay bounded.

Before this fix, ``_streaks`` accumulated one entry per thread_id that
ever hit ``_streak_bump`` via ``awrap_tool_call``. Runs that crashed
before ``aafter_agent`` could clear their entry leaked it forever. Over
a long-lived process with high crashy-traffic, the dict grew
unboundedly.

The fix adds FIFO eviction at a soft cap so steady-state growth is
pinned regardless of crash rate.
"""

from __future__ import annotations

import pytest

from langgraph_kit.core.resilience import loop_guard


@pytest.fixture(autouse=True)
def _isolate_streaks() -> None:
    """Each test gets a clean ``_streaks`` dict and the default cap."""
    loop_guard._streaks.clear()


def test_streak_bump_evicts_oldest_when_cap_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop_guard, "_STREAKS_SOFT_CAP", 3)

    loop_guard._streak_bump("thread-a", "tool_search")
    loop_guard._streak_bump("thread-b", "tool_search")
    loop_guard._streak_bump("thread-c", "tool_search")

    assert set(loop_guard._streaks.keys()) == {"thread-a", "thread-b", "thread-c"}

    # Next fresh thread should trigger eviction of the oldest ("thread-a").
    loop_guard._streak_bump("thread-d", "tool_search")

    assert set(loop_guard._streaks.keys()) == {"thread-b", "thread-c", "thread-d"}
    assert "thread-a" not in loop_guard._streaks


def test_streak_bump_does_not_evict_existing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-bumping an existing thread_id doesn't grow the dict, so no eviction."""
    monkeypatch.setattr(loop_guard, "_STREAKS_SOFT_CAP", 2)

    loop_guard._streak_bump("thread-a", "tool_search")
    loop_guard._streak_bump("thread-b", "tool_search")

    # Bumping thread-a again must not evict thread-b.
    loop_guard._streak_bump("thread-a", "tool_search")
    assert set(loop_guard._streaks.keys()) == {"thread-a", "thread-b"}


def test_streak_bump_stays_under_cap_under_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate many crashed runs that never got cleared."""
    monkeypatch.setattr(loop_guard, "_STREAKS_SOFT_CAP", 50)

    for i in range(500):
        loop_guard._streak_bump(f"thread-{i}", "tool_search")

    assert len(loop_guard._streaks) == 50
