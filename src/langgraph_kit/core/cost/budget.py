"""Store-backed budget manager for per-thread token limits."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langgraph_kit.core.cost.models import (
    BudgetCheckResult,
    BudgetConfig,
    BudgetState,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class BudgetManager:
    """Manages per-thread token budgets backed by the LangGraph Store.

    State is stored in namespace ``("budget", thread_id)`` with key ``"state"``.
    """

    def __init__(self, store: Any, config: BudgetConfig) -> None:
        super().__init__()
        self._store = store
        self._config = config

    async def load_state(self, thread_id: str) -> BudgetState:
        """Load the budget state for a thread, or return a fresh one."""
        try:
            # Direct aget — the prior ``asearch(..., limit=1)`` made the
            # retrieval fragile to anything else landing in the namespace
            # and added a Store scan where a single key lookup would do.
            item = await self._store.aget(("budget", thread_id), "state")
            if item is not None:
                return BudgetState.model_validate(item.value)
        except Exception:
            logger.warning(
                "Failed to load budget state for %s", thread_id, exc_info=True
            )
        return BudgetState(
            thread_id=thread_id, created_at=datetime.now(UTC).isoformat()
        )

    async def record_usage(
        self,
        thread_id: str,
        usage: TokenUsage,
        user_id: str = "",
    ) -> BudgetState:
        """Record token usage for a thread and persist updated state."""
        state = await self.load_state(thread_id)
        state.total_input_tokens += usage.input_tokens
        state.total_output_tokens += usage.output_tokens
        state.total_cost_usd += usage.estimated_cost_usd
        state.turn_count += 1
        state.updated_at = datetime.now(UTC).isoformat()
        if user_id:
            state.user_id = user_id

        try:
            await self._store.aput(
                ("budget", thread_id), "state", state.model_dump(mode="json")
            )
        except Exception:
            logger.warning(
                "Failed to save budget state for %s", thread_id, exc_info=True
            )

        return state

    async def check_budget(self, thread_id: str) -> BudgetCheckResult:
        """Check if the thread is within budget."""
        max_tokens = self._config.max_tokens_per_thread
        if max_tokens <= 0:
            return BudgetCheckResult(action="allow")

        state = await self.load_state(thread_id)
        total = state.total_input_tokens + state.total_output_tokens
        consumed_pct = total / max_tokens if max_tokens > 0 else 0.0
        remaining = max(0, max_tokens - total)

        if consumed_pct >= 1.0:
            return BudgetCheckResult(
                action="deny",
                reason=f"Token budget exhausted ({total}/{max_tokens} tokens used)",
                budget_consumed_pct=consumed_pct,
                remaining_tokens=0,
            )

        if (
            consumed_pct >= self._config.warning_threshold_pct
            and self._config.downgrade_model
        ):
            return BudgetCheckResult(
                action="downgrade",
                reason=f"Budget at {consumed_pct:.0%}, switching to {self._config.downgrade_model}",
                budget_consumed_pct=consumed_pct,
                remaining_tokens=remaining,
            )

        if consumed_pct >= self._config.warning_threshold_pct:
            return BudgetCheckResult(
                action="warn",
                reason=f"Budget at {consumed_pct:.0%} ({remaining} tokens remaining)",
                budget_consumed_pct=consumed_pct,
                remaining_tokens=remaining,
            )

        return BudgetCheckResult(
            action="allow",
            budget_consumed_pct=consumed_pct,
            remaining_tokens=remaining,
        )
