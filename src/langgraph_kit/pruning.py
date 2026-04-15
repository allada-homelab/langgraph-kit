"""Checkpoint and store data pruning utilities.

Provides functions to clean up old checkpoint data, tool result caches,
and stale queue items that accumulate over time.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Store namespaces that accumulate data
_TOOL_RESULTS_NAMESPACE = ("tool_results",)
_BUSY_NAMESPACE = ("thread_busy",)


class PruneResult(BaseModel):
    """Summary of a pruning operation."""

    tool_results_deleted: int = 0
    queue_items_deleted: int = 0
    stale_locks_cleared: int = 0


async def prune_store(
    store: Any,
    *,
    max_age_seconds: int = 604_800,  # 7 days
    dry_run: bool = False,
) -> PruneResult:
    """Remove stale data from the LangGraph Store.

    Cleans up:
      - Old tool result caches (namespace: ``tool_results``)
      - Stale busy-thread locks (namespace: ``thread_busy``)

    Args:
        store: LangGraph BaseStore instance (Postgres or in-memory).
        max_age_seconds: Delete items older than this (default: 7 days).
        dry_run: If True, count items but don't delete.

    Returns:
        PruneResult with counts of deleted items.
    """
    result = PruneResult()
    cutoff = time.time() - max_age_seconds

    # --- Prune tool result caches ---
    try:
        items = await store.asearch(_TOOL_RESULTS_NAMESPACE, limit=500)
        for item in items:
            created = item.value.get("created_at", 0) if hasattr(item, "value") else 0
            if created and created < cutoff:
                if not dry_run:
                    await store.adelete(_TOOL_RESULTS_NAMESPACE, item.key)
                result.tool_results_deleted += 1
    except Exception:
        logger.debug("Could not prune tool results", exc_info=True)

    # --- Clear stale busy locks ---
    try:
        items = await store.asearch(_BUSY_NAMESPACE, limit=500)
        for item in items:
            since = item.value.get("since", 0) if hasattr(item, "value") else 0
            if since and since < cutoff:
                if not dry_run:
                    await store.adelete(_BUSY_NAMESPACE, item.key)
                result.stale_locks_cleared += 1
    except Exception:
        logger.debug("Could not prune busy locks", exc_info=True)

    action = "Would delete" if dry_run else "Deleted"
    logger.info(
        "%s: %d tool results, %d stale locks",
        action,
        result.tool_results_deleted,
        result.stale_locks_cleared,
    )
    return result
