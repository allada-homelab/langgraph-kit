"""Coverage — write-time memory deduplication via ``find_duplicate``.

When ``embedding_fn`` is configured, ``PersistentMemoryManager`` can
detect that an incoming :class:`MemoryRecord` is semantically close
to an existing record in the same scope, and either skip the write
or raise depending on ``on_duplicate`` mode. Without
``embedding_fn`` the dedup probe is a no-op (no keyword fallback —
false positives at write time would silently merge distinct memories,
which is worse than missing a near-duplicate).
"""

from __future__ import annotations

import pytest

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import (
    DEFAULT_DEDUP_THRESHOLD,
    DuplicateMatch,
    DuplicateMemoryError,
    PersistentMemoryManager,
)
from tests.conftest import MockStore


def _record(
    *,
    title: str,
    summary: str = "",
    body: str = "",
    memory_type: MemoryType = MemoryType.USER,
    scope: MemoryScope = MemoryScope.USER,
) -> MemoryRecord:
    return MemoryRecord(
        title=title,
        summary=summary,
        body=body,
        type=memory_type,
        scope=scope,
    )


class _TagEmbedder:
    """Deterministic embedder over a small tag space.

    Each text gets a vector counting occurrences of the configured
    tags (case-insensitive). Two texts using the same tags yield the
    same direction → cosine similarity ~ 1.0. Predictable enough to
    pin specific similarity values in tests without floating-point
    fragility.
    """

    TAGS = ("python", "rust", "ruby")

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            low = t.lower()
            out.append([float(low.count(tag)) for tag in self.TAGS])
        return out


class TestFindDuplicateNoOpWithoutEmbedding:
    """No embedding fn → the probe is a no-op for safety."""

    async def test_returns_none_when_embedding_fn_unset(self) -> None:
        store = MockStore()
        mgr = PersistentMemoryManager(store)
        await mgr.create(_record(title="Python tips"))

        match = await mgr.find_duplicate(_record(title="Python tips"))
        assert match is None

    async def test_create_skip_writes_when_embedding_fn_unset(self) -> None:
        """Without embeddings, ``on_duplicate="skip"`` degrades to plain create."""
        store = MockStore()
        mgr = PersistentMemoryManager(store)
        first = _record(title="Python tips")
        second = _record(title="Python tips")  # textually identical

        await mgr.create(first)
        result = await mgr.create(second, on_duplicate="skip")
        assert isinstance(result, MemoryRecord)
        # Both records persisted; without embeddings we can't tell they're dupes.
        items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
        assert len(items) == 2


class TestFindDuplicate:
    """Embedding-backed similarity probe."""

    async def test_finds_match_above_threshold(self) -> None:
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        existing = _record(title="Likes Python", body="Python every day")
        await mgr.create(existing)

        candidate = _record(title="Python lover", body="Python python python")
        match = await mgr.find_duplicate(candidate)
        assert isinstance(match, DuplicateMatch)
        assert match.existing_id == existing.id
        assert match.similarity >= DEFAULT_DEDUP_THRESHOLD

    async def test_returns_none_below_threshold(self) -> None:
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        await mgr.create(_record(title="Likes Python", body="python python"))

        candidate = _record(title="Likes Rust", body="rust rust rust")
        match = await mgr.find_duplicate(candidate)
        assert match is None

    async def test_threshold_override(self) -> None:
        """Lowering the threshold lets a less-similar candidate count as dup."""
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        # Mixed-tag record: half python, half rust.
        await mgr.create(_record(title="Polyglot", body="python rust"))

        # Pure-python candidate: cosine vs (1, 1, 0) is 1/sqrt(2) ~ 0.707.
        candidate = _record(title="Pythonista", body="python python")
        assert await mgr.find_duplicate(candidate) is None  # default 0.92 too strict
        match = await mgr.find_duplicate(candidate, threshold=0.5)
        assert isinstance(match, DuplicateMatch)
        assert 0.5 <= match.similarity < DEFAULT_DEDUP_THRESHOLD

    async def test_per_scope_isolation(self) -> None:
        """Identical text in different scopes is *not* a duplicate."""
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        await mgr.create(_record(title="Python", body="python", scope=MemoryScope.USER))
        candidate = _record(title="Python", body="python", scope=MemoryScope.PROJECT)
        match = await mgr.find_duplicate(candidate)
        assert match is None


class TestCreateWithOnDuplicate:
    """``create(..., on_duplicate=...)`` honors the requested mode."""

    async def test_skip_returns_match_without_writing(self) -> None:
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        existing = _record(title="Likes Python", body="python python")
        await mgr.create(existing)

        candidate = _record(title="Python lover", body="python python python")
        result = await mgr.create(candidate, on_duplicate="skip")
        assert isinstance(result, DuplicateMatch)
        assert result.existing_id == existing.id

        items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
        assert len(items) == 1  # candidate not written

    async def test_create_mode_writes_unconditionally(self) -> None:
        """Default ``"create"`` preserves backwards compatibility."""
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        await mgr.create(_record(title="Likes Python", body="python python"))

        result = await mgr.create(
            _record(title="Likes Python", body="python python python")
        )
        assert isinstance(result, MemoryRecord)
        items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
        assert len(items) == 2

    async def test_raise_mode(self) -> None:
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())
        existing = _record(title="Likes Python", body="python python")
        await mgr.create(existing)

        candidate = _record(title="Python", body="python python python")
        with pytest.raises(DuplicateMemoryError) as excinfo:
            await mgr.create(candidate, on_duplicate="raise")
        assert excinfo.value.match.existing_id == existing.id

        # No record was written under the raise mode.
        items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
        assert len(items) == 1

    async def test_first_record_in_scope_writes_under_skip(self) -> None:
        """When there is nothing to dedup against, ``"skip"`` still writes."""
        store = MockStore()
        mgr = PersistentMemoryManager(store, embedding_fn=_TagEmbedder())

        result = await mgr.create(
            _record(title="Likes Python", body="python python"), on_duplicate="skip"
        )
        assert isinstance(result, MemoryRecord)
        items = await mgr.list_by_scope(MemoryScope.USER, MemoryType.USER)
        assert len(items) == 1
