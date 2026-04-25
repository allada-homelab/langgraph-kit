"""Memory consolidation: merge near-duplicates and prune stale records.

What this shows
---------------
- Seeding two nearly-duplicate memories under the same scope
- Wiring :class:`MemoryConsolidator` with a scripted JSON response so
  the demo runs hermetically (real consolidation calls the LLM to
  decide on ``keep`` / ``delete`` / ``merge`` / ``update``)
- Inspecting the :class:`ConsolidationResult` summary

The same consolidator is what the bundled scheduler invokes
periodically; here it's invoked synchronously so you can see the
state transitions inline.

How to run
----------
    uv run python -m examples.memory_consolidation

Expected output
---------------
    Seeded 2 near-duplicate memories under scope=USER.
    Result: ConsolidationResult(kept=0, deleted=0, merged=1, updated=0, errors=0)
    Final memories in scope: 1 record(s)
      - User wants concise, terse responses
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class _FakeAIResponse:
    """Minimal stand-in for ``AIMessage`` so consolidator's ``.content`` access works."""

    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedConsolidationLLM:
    """LLM stub for the consolidator. Ignores prompt, returns canned actions.

    Real consolidation calls ``llm.ainvoke([HumanMessage(...)], config=...)``;
    this stub returns whatever JSON we baked in so the demo's outcome is
    deterministic without touching a real model.
    """

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = actions

    async def ainvoke(self, _messages: Any, **_kwargs: Any) -> _FakeAIResponse:
        return _FakeAIResponse(json.dumps(self._actions))


from examples._lib import banner, line, make_in_memory_persistence  # noqa: E402


async def main() -> None:
    banner("memory_consolidation")

    from langgraph_kit.core.memory.consolidation import MemoryConsolidator
    from langgraph_kit.core.memory.models import (
        MemoryRecord,
        MemoryScope,
        MemoryType,
    )
    from langgraph_kit.core.memory.persistent import PersistentMemoryManager

    _, store = make_in_memory_persistence()
    mgr = PersistentMemoryManager(store)

    # 1. Seed two near-duplicates.
    duplicates = [
        MemoryRecord(
            title="User prefers terse responses",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="No trailing summaries.",
            body="The user finds recap paragraphs noisy.",
        ),
        MemoryRecord(
            title="User wants concise replies",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.USER,
            summary="Concise > verbose.",
            body="Skip the wrap-up; the diff is enough.",
        ),
    ]
    for rec in duplicates:
        await mgr.create(rec)
    line(f"Seeded {len(duplicates)} near-duplicate memories under scope=USER.")

    # 2. Script the LLM's consolidation directive: merge the two source
    #    records into one, dropping the originals.
    src_ids = [duplicates[0].id, duplicates[1].id]
    actions: list[dict[str, Any]] = [
        {
            "action": "merge",
            "source_ids": src_ids,
            "merged": {
                "title": "User wants concise, terse responses",
                "type": "feedback",
                "summary": "Skip wrap-ups; concise replies.",
                "body": "Combined feedback from two near-duplicate seed records.",
            },
            "reason": "near-duplicate feedback memories",
        }
    ]
    fake_llm = _ScriptedConsolidationLLM(actions)

    # 3. Run consolidation.
    consolidator = MemoryConsolidator(memory_manager=mgr, llm=fake_llm)
    result = await consolidator.consolidate(scope=MemoryScope.USER)
    line(f"Result: {result!r}")

    # 4. Inspect what survived.
    remaining = await mgr.list_by_scope(MemoryScope.USER)
    line(f"Final memories in scope: {len(remaining)} record(s)")
    for rec in remaining:
        line(f"  - {rec.title}")


if __name__ == "__main__":
    asyncio.run(main())
