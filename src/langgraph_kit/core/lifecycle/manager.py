"""Per-user data lifecycle: export, delete, anonymize.

Issue #31. The kit's Store namespaces are not all user-scoped today
(e.g. ``("memory", scope, type)`` is keyed by scope; ``("threads",
thread_id)`` is keyed by thread). A complete "delete every byte
about user X" pass requires walking every namespace, which in turn
requires knowing which user each row belongs to.

This module ships the *foundation* for that work: a manager surface
that today operates on the namespaces where ``user_id`` is already
present (audit log, threads metadata, optional ``user``-scoped
memory records). Coverage will widen as #33 (multi-tenancy) adds a
``tenant_id`` / ``user_id`` to the remaining namespaces.

Operations:

- :meth:`DataLifecycleManager.export` — returns a JSON-serializable
  dict ``{namespace: [items]}`` listing everything the kit currently
  knows about ``user_id``. Call sites should treat the keys as a
  contract that grows monotonically — older deployments may produce
  fewer keys than newer ones.
- :meth:`DataLifecycleManager.delete` — hard-deletes the same set.
  Returns the count removed. Each namespace removal is audited
  independently so a partial failure leaves a record of what was
  reached.
- :meth:`DataLifecycleManager.anonymize` — replaces identifying
  fields with a salted hash so the entry stays shape-compatible
  but the user identity is unrecoverable without the salt.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langgraph_kit.core.audit import AuditAction, AuditStore

logger = logging.getLogger(__name__)


def _pseudonym(user_id: str, salt: str) -> str:
    """Stable, non-reversible-without-salt pseudonym for ``user_id``.

    SHA-256 truncated to 16 hex chars (8 bytes) — long enough to avoid
    collision under realistic user counts, short enough to keep audit
    payloads scannable.
    """
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return f"anon-{digest[:16]}"


class DataLifecycleManager:
    """User-scoped export / delete / anonymize over the LangGraph Store."""

    def __init__(
        self,
        store: Any,
        audit: AuditStore | None = None,
        *,
        anonymize_salt: str = "lgk-default-salt",
    ) -> None:
        super().__init__()
        self._store = store
        self._audit = audit
        self._salt = anonymize_salt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _iter_threads_for_user(
        self, user_id: str
    ) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
        """Return ``(namespace, key, value)`` tuples for thread metadata
        belonging to ``user_id``.

        Walks the per-user thread index established by
        :class:`langgraph_kit.core.threads.ThreadManager`.
        """
        ns = ("thread_index", "by_user", user_id)
        try:
            items: list[Any] = await self._store.asearch(ns, limit=10_000)
        except Exception:
            logger.exception(
                "DataLifecycle: failed to list threads for user %s", user_id
            )
            return []
        return [(ns, item.key, item.value or {}) for item in items]

    async def _iter_audit_for_user(
        self, user_id: str
    ) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
        """Return audit entries whose actor is ``user:<user_id>``.

        Walks the per-month audit buckets the AuditStore writes to.
        Bounded — the kit's bucket pattern keeps any one read cheap.
        """
        # We don't have a global "list every audit bucket" call (the
        # AuditStore queries by date window). For a complete sweep we
        # check the last decade of buckets — same approach AuditStore
        # itself uses internally on an unbounded query.
        from datetime import UTC, datetime

        from langgraph_kit.core.audit.store import _month_buckets_descending

        end = datetime.now(UTC)
        start = end.replace(year=max(end.year - 10, 1))
        actor_key = f"user:{user_id}"
        out: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
        for bucket in _month_buckets_descending(start, end):
            ns = ("audit", bucket)
            try:
                items: list[Any] = await self._store.asearch(ns, limit=10_000)
            except Exception:
                logger.exception(
                    "DataLifecycle: failed to read audit bucket %s", bucket
                )
                continue
            for item in items:
                payload = item.value or {}
                if not isinstance(payload, dict):
                    continue
                if payload.get("actor") == actor_key:
                    out.append((ns, item.key, payload))
        return out

    async def _audit_event(
        self,
        user_id: str,
        action: AuditAction,
        target: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._audit is None:
            return
        try:
            await self._audit.write(
                actor=f"user:{user_id}",
                action=action,
                target=target,
                metadata=metadata,
            )
        except Exception:
            logger.exception("DataLifecycle: audit write failed for %s", user_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def export(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-serializable snapshot of *user_id*'s data."""
        threads = await self._iter_threads_for_user(user_id)
        audit = await self._iter_audit_for_user(user_id)

        export_payload: dict[str, list[dict[str, Any]]] = {
            "threads": [{"key": k, "value": v} for _, k, v in threads],
            "audit": [{"key": k, "value": v} for _, k, v in audit],
        }
        await self._audit_event(
            user_id,
            AuditAction.DATA_EXPORT,
            target=f"user:{user_id}",
            metadata={
                "thread_count": len(threads),
                "audit_count": len(audit),
            },
        )
        return export_payload

    async def delete(self, user_id: str) -> int:
        """Hard-delete every record the manager finds for ``user_id``."""
        threads = await self._iter_threads_for_user(user_id)
        audit = await self._iter_audit_for_user(user_id)

        removed = 0
        for ns, key, _val in threads:
            try:
                await self._store.adelete(ns, key)
                removed += 1
            except Exception:
                logger.exception("DataLifecycle: failed to delete %s/%s", ns, key)
        for ns, key, _val in audit:
            try:
                await self._store.adelete(ns, key)
                removed += 1
            except Exception:
                logger.exception("DataLifecycle: failed to delete %s/%s", ns, key)
        await self._audit_event(
            user_id,
            AuditAction.DATA_DELETE,
            target=f"user:{user_id}",
            metadata={
                "removed_count": removed,
                "thread_count": len(threads),
                "audit_count": len(audit),
            },
        )
        return removed

    async def anonymize(self, user_id: str) -> int:
        """Replace identifying fields with a salted pseudonym in place.

        Records survive — only the identifying field changes. Useful
        when the underlying data has analytical value (training,
        diagnostics) but the link to a real person must be severed.
        """
        pseudonym = _pseudonym(user_id, self._salt)
        threads = await self._iter_threads_for_user(user_id)
        audit = await self._iter_audit_for_user(user_id)

        rewritten = 0
        # Threads: re-key under the pseudonym index, drop the old row.
        for old_ns, key, value in threads:
            new_value = dict(value)
            new_value["user_id"] = pseudonym
            new_ns = ("thread_index", "by_user", pseudonym)
            try:
                await self._store.aput(new_ns, key, new_value)
                await self._store.adelete(old_ns, key)
                rewritten += 1
            except Exception:
                logger.exception(
                    "DataLifecycle: anonymize-thread failed for %s/%s", old_ns, key
                )

        # Audit entries: rewrite the actor field. Keep the original key
        # under the same bucket so timeline ordering is preserved.
        for ns, key, value in audit:
            new_value = dict(value)
            new_value["actor"] = f"user:{pseudonym}"
            try:
                await self._store.aput(ns, key, new_value)
                rewritten += 1
            except Exception:
                logger.exception(
                    "DataLifecycle: anonymize-audit failed for %s/%s", ns, key
                )

        await self._audit_event(
            user_id,
            # Anonymize is logged as a delete + new pseudonymous identity.
            AuditAction.DATA_DELETE,
            target=f"user:{user_id}",
            metadata={
                "rewritten_count": rewritten,
                "pseudonym": pseudonym,
                "kind": "anonymize",
            },
        )
        return rewritten
