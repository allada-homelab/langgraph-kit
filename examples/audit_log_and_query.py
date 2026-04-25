"""Audit log: write actions, query by actor / action / time window.

What this shows
---------------
- :class:`AuditStore.write` to append immutable rows describing
  ``(actor, action, target, metadata)``
- :class:`AuditStore.query` to filter by actor / action and time
  window — newest-first, bucketed monthly so listing is cheap

The audit log is intentionally append-only and best-effort: writes that
fail don't block the underlying action; reads return only what's
present. Used by HITL middleware, memory CRUD, and the security guards
to give compliance teams a forensic trail.

How to run
----------
    uv run python -m examples.audit_log_and_query

Expected output
---------------
    Wrote 3 audit entries.
    All entries (newest first):
      - agent_invoke    by agent:reference-deep-agent on thread:abc-123
      - memory_create   by agent:reference-deep-agent on memory:user_pref
      - injection_detected by system on thread:abc-123
    Filtered by action=memory_create:
      - memory_create   by agent:reference-deep-agent on memory:user_pref
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, line, make_in_memory_persistence


async def main() -> None:
    banner("audit_log_and_query")

    from langgraph_kit.core.audit import AuditAction, AuditStore

    _, store = make_in_memory_persistence()
    audit = AuditStore(store)

    # 1. Write three entries. The store is best-effort: a failing write
    #    is logged and swallowed so audit never blocks real work.
    await audit.write(
        actor="agent:reference-deep-agent",
        action=AuditAction.AGENT_INVOKE,
        target="thread:abc-123",
        metadata={"user_id": "u-42"},
    )
    await audit.write(
        actor="agent:reference-deep-agent",
        action=AuditAction.MEMORY_CREATE,
        target="memory:user_pref",
        metadata={"title": "User prefers terse responses"},
    )
    await audit.write(
        actor="system",
        action=AuditAction.INJECTION_DETECTED,
        target="thread:abc-123",
        metadata={"patterns": ["ignore_previous_instructions"]},
    )
    line("Wrote 3 audit entries.")

    # 2. Query: no filter → newest-first by timestamp.
    line("All entries (newest first):")
    for entry in await audit.query(limit=10):
        line(f"  - {entry.action.value:<22} by {entry.actor} on {entry.target}")

    # 3. Filter by action.
    line("Filtered by action=memory_create:")
    for entry in await audit.query(action=AuditAction.MEMORY_CREATE):
        line(f"  - {entry.action.value:<22} by {entry.actor} on {entry.target}")


if __name__ == "__main__":
    asyncio.run(main())
