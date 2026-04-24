"""Store-backed trace persistence."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.tracing.models import TraceRecord, TraceSummary

logger = logging.getLogger(__name__)

# Summary key suffix. Traces themselves live at ``trace_id``; summaries
# live at ``trace_id + _SUMMARY_SUFFIX``. list_traces reads only the
# summaries to avoid materialising every span tree just to render a list.
_SUMMARY_SUFFIX = ".summary"


class TraceStore:
    """Persists execution traces in the LangGraph Store.

    Traces are stored under namespace ``("traces", thread_id)`` keyed by
    ``trace_id``. A companion ``trace_id + ".summary"`` key holds a
    ``TraceSummary`` so ``list_traces`` can avoid loading full span trees.
    """

    def __init__(self, store: Any, *, max_per_thread: int = 20) -> None:
        super().__init__()
        self._store = store
        self._max_per_thread = max_per_thread

    async def save_trace(self, thread_id: str, trace: TraceRecord) -> None:
        """Save a trace (and its summary) and prune old ones if over the limit."""
        try:
            namespace = ("traces", thread_id)
            await self._store.aput(
                namespace,
                trace.trace_id,
                trace.model_dump(mode="json"),
            )
            summary = TraceSummary(
                trace_id=trace.trace_id,
                started_at=trace.started_at,
                duration_ms=trace.duration_ms,
                span_count=trace.span_count,
            )
            await self._store.aput(
                namespace,
                trace.trace_id + _SUMMARY_SUFFIX,
                summary.model_dump(mode="json"),
            )
            await self._prune(thread_id)
        except Exception:
            logger.warning(
                "Failed to save trace for thread %s", thread_id, exc_info=True
            )

    async def get_trace(self, thread_id: str, trace_id: str) -> TraceRecord | None:
        """Get a single trace by ID using a direct ``aget`` round-trip."""
        try:
            item = await self._store.aget(("traces", thread_id), trace_id)
            if item is not None:
                return TraceRecord.model_validate(item.value)
        except Exception:
            logger.warning(
                "Failed to get trace %s/%s", thread_id, trace_id, exc_info=True
            )
        return None

    async def list_traces(self, thread_id: str) -> list[TraceSummary]:
        """List trace summaries for a thread.

        Reads the materialised ``.summary`` keys to avoid deserialising
        every full span tree just to show a list.
        """
        try:
            items = await self._store.asearch(("traces", thread_id), limit=200)
            summaries: list[TraceSummary] = []
            for item in items:
                if not item.key.endswith(_SUMMARY_SUFFIX):
                    continue
                summaries.append(TraceSummary.model_validate(item.value))
            summaries.sort(key=lambda s: s.started_at, reverse=True)
            return summaries
        except Exception:
            logger.warning(
                "Failed to list traces for thread %s", thread_id, exc_info=True
            )
            return []

    async def _prune(self, thread_id: str) -> None:
        """Remove oldest traces (and their summaries) beyond the max limit."""
        try:
            items = await self._store.asearch(("traces", thread_id), limit=500)
            # Only count full-trace keys for the cap — summaries are
            # companion rows.
            trace_items = [
                i for i in items if not i.key.endswith(_SUMMARY_SUFFIX)
            ]
            if len(trace_items) <= self._max_per_thread:
                return

            sorted_items = sorted(
                trace_items, key=lambda i: i.value.get("started_at", "")
            )
            to_remove = sorted_items[: len(sorted_items) - self._max_per_thread]
            namespace = ("traces", thread_id)
            for item in to_remove:
                await self._store.adelete(namespace, item.key)
                await self._store.adelete(namespace, item.key + _SUMMARY_SUFFIX)
        except Exception:
            logger.warning(
                "Failed to prune traces for thread %s", thread_id, exc_info=True
            )
