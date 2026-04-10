"""Store-backed trace persistence."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.tracing.models import TraceRecord, TraceSummary

logger = logging.getLogger(__name__)


class TraceStore:
    """Persists execution traces in the LangGraph Store.

    Traces are stored under namespace ``("traces", thread_id)`` keyed by trace_id.
    """

    def __init__(self, store: Any, *, max_per_thread: int = 20) -> None:
        self._store = store
        self._max_per_thread = max_per_thread

    async def save_trace(self, thread_id: str, trace: TraceRecord) -> None:
        """Save a trace and prune old ones if over the limit."""
        try:
            await self._store.aput(
                ("traces", thread_id),
                trace.trace_id,
                trace.model_dump(mode="json"),
            )
            # Prune old traces
            await self._prune(thread_id)
        except Exception:
            logger.debug("Failed to save trace for thread %s", thread_id, exc_info=True)

    async def get_trace(self, thread_id: str, trace_id: str) -> TraceRecord | None:
        """Get a single trace by ID."""
        try:
            items = await self._store.asearch(("traces", thread_id), limit=100)
            for item in items:
                if item.key == trace_id:
                    return TraceRecord.model_validate(item.value)
        except Exception:
            logger.debug("Failed to get trace %s/%s", thread_id, trace_id, exc_info=True)
        return None

    async def list_traces(self, thread_id: str) -> list[TraceSummary]:
        """List trace summaries for a thread."""
        try:
            items = await self._store.asearch(("traces", thread_id), limit=100)
            summaries = []
            for item in items:
                data = item.value
                record = TraceRecord.model_validate(data)
                summaries.append(
                    TraceSummary(
                        trace_id=record.trace_id,
                        started_at=record.started_at,
                        duration_ms=record.duration_ms,
                        span_count=record.span_count,
                    )
                )
            # Sort by started_at descending
            summaries.sort(key=lambda s: s.started_at, reverse=True)
            return summaries
        except Exception:
            logger.debug("Failed to list traces for thread %s", thread_id, exc_info=True)
            return []

    async def _prune(self, thread_id: str) -> None:
        """Remove oldest traces beyond the max limit."""
        try:
            items = await self._store.asearch(("traces", thread_id), limit=200)
            if len(items) <= self._max_per_thread:
                return

            # Sort by started_at and remove oldest
            sorted_items = sorted(items, key=lambda i: i.value.get("started_at", ""))
            to_remove = sorted_items[: len(sorted_items) - self._max_per_thread]
            for item in to_remove:
                await self._store.adelete(("traces", thread_id), item.key)
        except Exception:
            logger.debug("Failed to prune traces for thread %s", thread_id, exc_info=True)
