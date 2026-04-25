"""Memory: typed, scoped persistent records with CRUD + search.

What this shows
---------------
- Creating a few typed memory records (``MemoryRecord`` + ``MemoryType`` + ``MemoryScope``)
- Listing them by scope, fetching by id, and updating in place
- Keyword search (semantic search is shown in Phase 3, gated on
  ``AgentConfig.memory_embedding_fn``)

No LLM is used — this is pure store-backed CRUD. The same
:class:`PersistentMemoryManager` is what the auto-extraction middleware
populates after each agent turn.

How to run
----------
    uv run python -m examples.memory_save_recall

Expected output
---------------
    Created 3 memory record(s).
    Listing FEEDBACK records in USER scope (limit=5):
      - User prefers terse responses
      - User wants pytest assertions on separate lines
      - ...
    Search hit for 'terse': User prefers terse responses
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, line, make_in_memory_persistence


async def main() -> None:
    banner("memory_save_recall")

    from langgraph_kit.core.memory.models import (
        MemoryRecord,
        MemoryScope,
        MemoryType,
    )
    from langgraph_kit.core.memory.persistent import PersistentMemoryManager

    _, store = make_in_memory_persistence()
    mgr = PersistentMemoryManager(store)

    # 1. Create three feedback records.
    seeds = [
        MemoryRecord(
            title="User prefers terse responses",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="No trailing summaries; the diff is sufficient.",
            body="When closing a task, omit the recap paragraph.",
        ),
        MemoryRecord(
            title="User wants pytest assertions on separate lines",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="Avoid `assert a and b` — split into two asserts.",
            body="Failure messages are clearer when each clause is its own assert.",
        ),
        MemoryRecord(
            title="Project tracker is GitHub Project #1",
            type=MemoryType.PROJECT,
            scope=MemoryScope.PROJECT,
            summary="https://github.com/orgs/allada-homelab/projects/1",
            body="All langgraph-kit feature work is tracked there.",
        ),
    ]
    for record in seeds:
        await mgr.create(record)
    line(f"Created {len(seeds)} memory record(s).")

    # 2. List by scope + type.
    line("Listing FEEDBACK records in USER scope (limit=5):")
    listing = await mgr.list_by_scope(
        MemoryScope.USER, memory_type=MemoryType.FEEDBACK, limit=5
    )
    for rec in listing:
        line(f"  - {rec.title}")

    # 3. Round-trip an update.
    target = listing[0]
    updated = await mgr.update(
        target.id,
        target.scope,
        {"summary": "(updated) No trailing summaries; the diff is sufficient."},
    )
    assert updated is not None
    line(f"Updated record: {updated.summary}")

    # 4. Keyword search — no embedding function configured, so this
    # falls back to case-insensitive token overlap on title + summary + body.
    query = "terse"
    hits = await mgr.search(query, scope=MemoryScope.USER, limit=1)
    if hits:
        line(f"Search hit for {query!r}: {hits[0].title}")
    else:
        line(f"No search hits for {query!r}.")


if __name__ == "__main__":
    asyncio.run(main())
