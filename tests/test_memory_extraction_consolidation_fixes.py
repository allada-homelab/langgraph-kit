"""Regression tests for Phase E memory-extraction and -consolidation fixes.

Covers:
- ``AutoMemoryExtractor`` honours per-candidate ``scope``.
- ``AutoMemoryExtractor`` passes ``type`` through on ``update`` actions.
- ``AutoMemoryExtractor`` caps the number of applied candidates.
- ``MemoryConsolidator`` merge path creates new record before deleting sources.
- ``MemoryConsolidator`` update path rejects non-whitelisted fields.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.memory.consolidation import MemoryConsolidator
from langgraph_kit.core.memory.extraction import AutoMemoryExtractor
from langgraph_kit.core.memory.models import MemoryRecord, MemoryScope, MemoryType
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

from .conftest import MockStore


class _FakeLLM:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    async def ainvoke(self, messages: list[Any], config: Any = None) -> Any:
        class _Resp:
            content = self._raw

        return _Resp()


@pytest.mark.asyncio
async def test_extractor_honors_candidate_scope() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    raw = (
        '[{"action": "create", "title": "Project lead", "type": "project",'
        ' "scope": "project", "summary": "s", "body": "b"}]'
    )
    extractor = AutoMemoryExtractor(mgr, _FakeLLM(raw))

    results = await extractor.extract(
        recent_messages=[_msg("human", "hi")],
        scope=MemoryScope.USER,  # caller default — must be overridden.
    )
    assert len(results) == 1
    assert results[0].scope == MemoryScope.PROJECT


@pytest.mark.asyncio
async def test_extractor_caps_candidates(monkeypatch: Any) -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    # 20 valid create candidates.
    parts = [
        f'{{"action": "create", "title": "t{i}", "type": "user",'
        f' "scope": "user", "summary": "s", "body": "b{i}"}}'
        for i in range(20)
    ]
    raw = "[" + ",".join(parts) + "]"
    extractor = AutoMemoryExtractor(mgr, _FakeLLM(raw), max_candidates=3)

    results = await extractor.extract(
        recent_messages=[_msg("human", "msg")], scope=MemoryScope.USER
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_extractor_update_passes_type_through() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    seed = MemoryRecord(
        id="mem-42",
        title="Orig",
        type=MemoryType.USER,
        scope=MemoryScope.USER,
        summary="s",
        body="b",
    )
    await mgr.create(seed)

    raw = (
        '[{"action": "update", "id": "mem-42", "title": "New",'
        ' "type": "feedback", "summary": "s2", "body": "b2"}]'
    )
    extractor = AutoMemoryExtractor(mgr, _FakeLLM(raw))

    results = await extractor.extract(
        recent_messages=[_msg("human", "msg")], scope=MemoryScope.USER
    )
    assert len(results) == 1
    assert results[0].type == MemoryType.FEEDBACK


@pytest.mark.asyncio
async def test_consolidator_merge_creates_before_deleting(
    monkeypatch: Any,
) -> None:
    """Crash between create and delete should leave merged record intact,
    not wipe source data with nothing to show for it."""
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    # Seed two records that will be merged.
    for i in (1, 2):
        await mgr.create(
            MemoryRecord(
                id=f"src-{i}",
                title=f"Src {i}",
                type=MemoryType.USER,
                scope=MemoryScope.USER,
                summary="s",
                body=f"b{i}",
            )
        )

    raw = (
        '[{"action": "merge", "source_ids": ["src-1", "src-2"],'
        ' "merged": {"title": "Merged", "type": "user", "summary": "sm",'
        ' "body": "bm"}}]'
    )

    # Patch delete to simulate a crash immediately after the first delete.
    real_delete = mgr.delete
    call_count = {"n": 0}

    async def _flaky_delete(record_id: str, scope: MemoryScope, **kwargs: Any) -> bool:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated crash")
        return await real_delete(record_id, scope, **kwargs)

    consolidator = MemoryConsolidator(mgr, _FakeLLM(raw))
    # Swap in the flaky delete only on the second apply step.
    monkeypatch.setattr(mgr, "delete", _flaky_delete)
    result = await consolidator.consolidate(scope=MemoryScope.USER)

    # The merged record should exist regardless of the crash — if we had
    # deleted sources first, a crash between delete and create would wipe
    # data outright.
    all_records = await mgr.list_by_scope(MemoryScope.USER, limit=100)
    merged_records = [r for r in all_records if r.title == "Merged"]
    assert merged_records, (
        "Merged record should have been created before any delete attempt."
    )
    # No specific assertion on source records — delete may have partially
    # completed. The invariant is "merged exists".
    _ = result


@pytest.mark.asyncio
async def test_consolidator_update_rejects_unsafe_fields() -> None:
    """LLM suggesting ``updates={"id": ..., "scope": ...}`` must not take effect."""
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    await mgr.create(
        MemoryRecord(
            id="mem-42",
            title="Original",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
            summary="orig-summary",
            body="orig-body",
        )
    )
    # Consolidator skips when <2 records exist in the scope.
    await mgr.create(
        MemoryRecord(
            id="mem-43",
            title="Filler",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
            summary="s",
            body="b",
        )
    )

    raw = (
        '[{"action": "update", "id": "mem-42", "updates":'
        ' {"id": "hijacked", "scope": "team", "title": "New"}}]'
    )

    consolidator = MemoryConsolidator(mgr, _FakeLLM(raw))
    await consolidator.consolidate(scope=MemoryScope.USER)

    # The title update should have been applied; id/scope must not have.
    out = await mgr.get("mem-42", MemoryScope.USER)
    assert out is not None, "Record id must not have been rewritten."
    assert out.title == "New"
    assert out.scope == MemoryScope.USER, "Scope must not have been rewritten."


def _msg(role: str, content: str) -> Any:
    class _M:
        type = role

        def __init__(self, c: str) -> None:
            self.content = c

    return _M(content)


class _TagEmbedder:
    """Tag-counting embedder for write-time dedup tests."""

    TAGS = ("python", "rust")

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(t.lower().count(tag)) for tag in self.TAGS] for t in texts]


@pytest.mark.asyncio
async def test_extractor_skips_near_duplicate_writes() -> None:
    """When ``embedding_fn`` is configured, the extractor's create
    path must not write near-duplicates.

    Regression test for #9: previously dedup was prompt-only and the
    LLM occasionally produced duplicates that got persisted.
    """
    store = MockStore()
    mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
    # Seed an existing record about Python.
    await mgr.create(
        MemoryRecord(
            title="Likes Python",
            summary="enjoys python",
            body="python python python",
            type=MemoryType.USER,
            scope=MemoryScope.USER,
        )
    )

    # LLM proposes another Python record — should be detected as a dup.
    raw = (
        '[{"action": "create", "title": "Python fan", "type": "user",'
        ' "scope": "user", "summary": "loves python",'
        ' "body": "python python python"}]'
    )
    extractor = AutoMemoryExtractor(mgr, _FakeLLM(raw))

    results = await extractor.extract(
        recent_messages=[_msg("human", "I love python")], scope=MemoryScope.USER
    )

    # Duplicate suppressed → no record returned by the extractor, and
    # only the original survives in the store.
    assert results == []
    items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
    assert len(items) == 1
    assert items[0].title == "Likes Python"
