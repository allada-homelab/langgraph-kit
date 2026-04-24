"""Regression: TokenTrackingCallback totals must reach BudgetManager Store.

Before this fix the SSE budget event fired but nothing persisted the
per-thread totals. The next turn's ``check_budget`` then saw a stale
0-token state and never denied, regardless of how much had actually
been consumed.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit._config import AgentConfig, configure
from langgraph_kit.core.cost.budget import BudgetManager
from langgraph_kit.core.cost.callback import TokenTrackingCallback
from langgraph_kit.core.cost.models import BudgetConfig, TokenUsage
from langgraph_kit.streaming import stream_agent_events

from .conftest import MockStore


class _QuickGraph:
    """Graph that yields nothing and returns an empty state — lets the
    finally block run without any event-loop work."""

    async def astream_events(
        self, input_data: Any, config: Any = None, version: str = "v2"
    ) -> Any:
        if False:
            yield None  # pragma: no cover

    async def aget_state(self, config: Any) -> Any:
        class _S:
            values: dict[str, Any] = {}  # noqa: RUF012
            tasks: list[Any] = []  # noqa: RUF012

        return _S()

    config: Any = None


@pytest.mark.asyncio
async def test_stream_persists_budget_usage(monkeypatch: Any) -> None:
    store = MockStore()

    # Simulate the callback having accumulated one turn's usage.
    callback = TokenTrackingCallback()
    callback._accumulated.append(
        TokenUsage(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            model="gpt-4o-mini",
            estimated_cost_usd=0.0001,
        )
    )

    # The kit's budget-tracking gate is ``token_budget_per_thread > 0``.
    configure(AgentConfig(token_budget_per_thread=10_000))

    config = {
        "configurable": {"thread_id": "tid-budget"},
        "metadata": {
            "_budget_callback": callback,
            "user_id": "u-1",
        },
    }

    async for _ in stream_agent_events(
        _QuickGraph(), {"messages": []}, config, store=store
    ):
        pass

    # BudgetManager should now see the persisted state.
    manager = BudgetManager(
        store, BudgetConfig(max_tokens_per_thread=10_000)
    )
    state = await manager.load_state("tid-budget")
    assert state.total_input_tokens == 500
    assert state.total_output_tokens == 200
    assert state.user_id == "u-1"

    # Restore baseline config so other tests aren't affected.
    configure(AgentConfig())
