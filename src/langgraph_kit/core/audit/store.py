"""Append-only audit log over a LangGraph ``BaseStore``.

The :class:`AuditStore` is the only sanctioned writer and reader for
audit data. Every audit producer (security middleware, memory CRUD,
HITL hooks, FastAPI lifespan) calls :meth:`write` rather than poking
the store directly, so future additions like signature chaining or
external sinks land in one place.

Storage layout: entries are written to the namespace
``("audit", <YYYY_MM>)`` keyed by ``entry.id``. Bucket-by-month keeps
month-range listings to a single namespace scan instead of a global
table walk; multi-month queries iterate buckets in reverse order so
the common "give me the last N entries" path stops as soon as it has
enough.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from .models import AuditAction, AuditEntry

logger = logging.getLogger(__name__)


_AUDIT_NS_PREFIX: str = "audit"


def _bucket_namespace(bucket_key: str) -> tuple[str, str]:
    return (_AUDIT_NS_PREFIX, bucket_key)


def _month_buckets_descending(start: datetime, end: datetime) -> list[str]:
    """Year-month bucket keys covering [start, end] in reverse order.

    Both endpoints are inclusive at month granularity.
    """
    if start > end:
        start, end = end, start
    cursor = datetime(end.year, end.month, 1, tzinfo=UTC)
    floor = datetime(start.year, start.month, 1, tzinfo=UTC)
    out: list[str] = []
    while cursor >= floor:
        out.append(f"{cursor.year:04d}_{cursor.month:02d}")
        # Step back one month.
        if cursor.month == 1:
            cursor = cursor.replace(year=cursor.year - 1, month=12)
        else:
            cursor = cursor.replace(month=cursor.month - 1)
    return out


class AuditStore:
    """Sanctioned read/write surface for the audit log."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self._store = store

    async def write(
        self,
        *,
        actor: str,
        action: AuditAction,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new audit entry. Returns the persisted record."""
        entry = AuditEntry(
            actor=actor,
            action=action,
            target=target,
            metadata=metadata or {},
        )
        try:
            await self._store.aput(
                _bucket_namespace(entry.bucket_key()),
                entry.id,
                entry.model_dump(mode="json"),
            )
        except Exception:
            # Audit must never block a real action. Log and swallow.
            logger.exception(
                "AuditStore.write failed: actor=%s action=%s target=%s",
                actor,
                action.value,
                target,
            )
        return entry

    async def query(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        actor: str | None = None,
        action: AuditAction | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Return matching entries, newest-first.

        All filters are AND'd. ``since`` defaults to the earliest
        timestamp in the store (effectively unbounded); ``until``
        defaults to ``now``. Returned list is at most ``limit`` long.
        Buckets are scanned newest-first so the common case
        ("last N entries") is cheap.
        """
        if limit <= 0:
            return []
        end_dt = until if until is not None else datetime.now(UTC)
        # Default ``since`` to a wide window — without a real lower bound
        # we'd have to know every bucket the store has ever held, which
        # we don't. 10 years is arbitrary but cheap (120 buckets).
        start_dt = (
            since
            if since is not None
            else end_dt.replace(year=max(end_dt.year - 10, 1))
        )

        results: list[AuditEntry] = []
        for bucket in _month_buckets_descending(start_dt, end_dt):
            try:
                items: list[Any] = await self._store.asearch(
                    _bucket_namespace(bucket), limit=10_000
                )
            except Exception:
                logger.exception("AuditStore.query failed on bucket %s", bucket)
                continue
            entries: list[AuditEntry] = []
            for item in items:
                payload = item.value
                if not isinstance(payload, dict):
                    continue
                try:
                    entry = AuditEntry.model_validate(payload)
                except Exception:
                    logger.exception(
                        "AuditStore: invalid entry %s in bucket %s",
                        item.key,
                        bucket,
                    )
                    continue
                if entry.timestamp < start_dt or entry.timestamp > end_dt:
                    continue
                if actor is not None and entry.actor != actor:
                    continue
                if action is not None and entry.action != action:
                    continue
                if target is not None and entry.target != target:
                    continue
                entries.append(entry)
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            for entry in entries:
                results.append(entry)
                if len(results) >= limit:
                    return results
        return results
