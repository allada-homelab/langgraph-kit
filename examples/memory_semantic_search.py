"""Memory: semantic search via a user-supplied embedding function.

What this shows
---------------
- Wiring :class:`PersistentMemoryManager` with an ``embedding_fn`` so
  ``search()`` ranks by cosine similarity instead of token overlap
- The kit's design choice: the *presence* of the callable is the
  switch — there's no silent fallback from semantic to keyword, so
  behaviour is deterministic per build
- A toy bag-of-words embedding function so the demo runs hermetically;
  swap in a real embedding API for production

How to run
----------
    uv run python -m examples.memory_semantic_search

Expected output
---------------
    Created 4 memory record(s).
    Semantic search for 'how to ship things' (top 1):
      - Always run just pre-commit before push
"""

from __future__ import annotations

import asyncio
import re

from examples._lib import banner, line, make_in_memory_persistence


def _toy_embedding_for_text(text: str) -> list[float]:
    """Bag-of-words count vector over a fixed vocabulary.

    Real semantic search uses a model-served embedding; this toy
    keeps the demo hermetic. The vocabulary is curated so the seed
    records and the demo query produce non-trivial similarities.
    """
    vocab = [
        "user",
        "terse",
        "pytest",
        "assertions",
        "split",
        "lines",
        "github",
        "project",
        "tracker",
        "commit",
        "pre",
        "push",
        "branch",
        "ship",
    ]
    tokens = {m.group(0).lower() for m in re.finditer(r"\w+", text)}
    return [1.0 if word in tokens else 0.0 for word in vocab]


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Async batch embedding wrapper that the kit's API expects."""
    return [_toy_embedding_for_text(t) for t in texts]


async def main() -> None:
    banner("memory_semantic_search")

    from langgraph_kit.core.memory.models import (
        MemoryRecord,
        MemoryScope,
        MemoryType,
    )
    from langgraph_kit.core.memory.persistent import PersistentMemoryManager

    _, store = make_in_memory_persistence()
    mgr = PersistentMemoryManager(store, embedding_fn=_embed_batch)

    seeds = [
        MemoryRecord(
            title="User prefers terse responses",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="No trailing summaries.",
            body="terse responses",
        ),
        MemoryRecord(
            title="Split assertions onto separate lines",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="Split assertions for clearer pytest failures.",
            body="pytest assertions split lines",
        ),
        MemoryRecord(
            title="Always run just pre-commit before push",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="Catch lints locally before they hit CI.",
            body="pre-commit before push to ship cleanly",
        ),
        MemoryRecord(
            title="Project tracker is GitHub Project #1",
            type=MemoryType.PROJECT,
            scope=MemoryScope.PROJECT,
            summary="GitHub project tracker for langgraph-kit.",
            body="github project tracker",
        ),
    ]
    for rec in seeds:
        await mgr.create(rec)
    line(f"Created {len(seeds)} memory record(s).")

    # Semantic-search query — overlapping vocab with the "ship" / "push"
    # records ranks them above the unrelated entries.
    query = "how to ship things"
    hits = await mgr.search(query, scope=MemoryScope.USER, limit=2)
    line(f"Semantic search for {query!r} (top {len(hits)}):")
    for rec in hits:
        line(f"  - {rec.title}")


if __name__ == "__main__":
    asyncio.run(main())
