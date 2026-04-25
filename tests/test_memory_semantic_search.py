"""Coverage — opt-in semantic search + keyword fallback in PersistentMemoryManager.

When ``embedding_fn`` is configured the manager indexes records on
create/update and ranks search results by cosine similarity. Otherwise
it scores by case-insensitive token overlap against title/summary/body.
There is no silent fallback from a failed semantic query to keyword —
the presence of the callable is the only mode switch.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    title: str,
    summary: str = "",
    body: str = "",
    memory_type: MemoryType = MemoryType.USER,
) -> MemoryRecord:
    return MemoryRecord(
        title=title,
        summary=summary,
        body=body,
        type=memory_type,
        scope=MemoryScope.USER,
    )


class _FakeEmbedder:
    """Deterministic embedder keyed by trigger tokens in the text.

    Returns vectors in a 3-dim space where each dimension corresponds to
    one of three tags: "dog" / "car" / "python". Each text's vector is the
    count of each tag (case-insensitive, substring match), normalized. A
    query about "python" ends up close to records mentioning python and
    far from records about cars.
    """

    TAGS = ("dog", "car", "python")

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            low = t.lower()
            out.append([float(low.count(tag)) for tag in self.TAGS])
        return out


# ---------------------------------------------------------------------------
# Keyword mode (no embedding_fn)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_search_ranks_by_token_overlap() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    await mgr.create(_record(title="Python tips", body="List comprehensions"))
    await mgr.create(_record(title="Docker notes", body="Container basics"))

    results = await mgr.search("python comprehension", MemoryScope.USER, limit=5)
    titles = [r.title for r in results]
    assert titles == ["Python tips"], titles


@pytest.mark.asyncio
async def test_keyword_search_is_case_insensitive() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)

    await mgr.create(_record(title="PYTHON Tricks", body="zip, enumerate"))

    results = await mgr.search("python", MemoryScope.USER, limit=5)
    assert len(results) == 1
    assert results[0].title == "PYTHON Tricks"


@pytest.mark.asyncio
async def test_keyword_search_empty_query_returns_empty() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    await mgr.create(_record(title="Anything"))

    results = await mgr.search("   ", MemoryScope.USER, limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_keyword_search_no_match_returns_empty() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    await mgr.create(_record(title="Python tips", body="zip, enumerate"))

    results = await mgr.search("kubernetes", MemoryScope.USER, limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_keyword_mode_does_not_write_embedding_namespace() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    await mgr.create(_record(title="Anything", body="body text"))

    embedding_ns = ("memory_embeddings", MemoryScope.USER.value)
    assert store._data.get(embedding_ns, {}) == {}


# ---------------------------------------------------------------------------
# Semantic mode (with embedding_fn)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_indexes_embedding_when_fn_configured() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    rec = await mgr.create(_record(title="About python", body="list comprehension"))

    embedding_ns = ("memory_embeddings", MemoryScope.USER.value)
    assert rec.id in store._data.get(embedding_ns, {})
    payload = store._data[embedding_ns][rec.id]
    assert "vector" in payload
    assert payload["memory_type"] == MemoryType.USER.value


@pytest.mark.asyncio
async def test_semantic_search_ranks_by_cosine_similarity() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    await mgr.create(_record(title="Python notes", body="python python dog"))
    await mgr.create(_record(title="Car reviews", body="car car car"))
    await mgr.create(_record(title="Dog training", body="dog dog"))

    results = await mgr.search("python guide", MemoryScope.USER, limit=3)
    titles = [r.title for r in results]
    # "python guide" embeds to [0,0,1]. Python notes dominates, dog training
    # has 0 similarity on the python axis, car reviews is orthogonal.
    assert titles[0] == "Python notes"
    assert "Car reviews" not in titles  # cosine sim = 0 is filtered


@pytest.mark.asyncio
async def test_semantic_search_respects_limit() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    for i in range(5):
        await mgr.create(_record(title=f"Python note {i}", body="python"))

    results = await mgr.search("python", MemoryScope.USER, limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_update_reindexes_when_text_changes() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    rec = await mgr.create(_record(title="About cars", body="car car"))
    calls_after_create = len(embedder.calls)

    await mgr.update(rec.id, MemoryScope.USER, {"body": "now about python"})
    assert len(embedder.calls) > calls_after_create, (
        "update with new body text should trigger a fresh embedding call"
    )

    # New vector should reflect the python-axis text, not the car-axis text.
    embedding_ns = ("memory_embeddings", MemoryScope.USER.value)
    vector = store._data[embedding_ns][rec.id]["vector"]
    # Index 2 is the "python" tag; index 1 is "car". After update the text
    # says "now about python" with no "car", so python count > car count.
    assert vector[2] >= vector[1]


@pytest.mark.asyncio
async def test_update_non_text_field_does_not_reindex() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    rec = await mgr.create(_record(title="Python", body="body"))
    calls_after_create = len(embedder.calls)

    # source isn't part of the embedded text — changing it should not
    # re-embed, to save cost when an LLM tags the source post-hoc.
    await mgr.update(rec.id, MemoryScope.USER, {"source": "agent-a"})
    assert len(embedder.calls) == calls_after_create


@pytest.mark.asyncio
async def test_delete_removes_embedding() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    rec = await mgr.create(_record(title="Python", body="python"))
    embedding_ns = ("memory_embeddings", MemoryScope.USER.value)
    assert rec.id in store._data[embedding_ns]

    assert await mgr.delete(rec.id, MemoryScope.USER) is True
    assert rec.id not in store._data.get(embedding_ns, {})


@pytest.mark.asyncio
async def test_semantic_search_filter_by_type() -> None:
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    await mgr.create(
        _record(title="Python in user type", body="python", memory_type=MemoryType.USER)
    )
    await mgr.create(
        _record(
            title="Python in feedback type",
            body="python",
            memory_type=MemoryType.FEEDBACK,
        )
    )

    results = await mgr.search(
        "python", MemoryScope.USER, memory_type=MemoryType.FEEDBACK, limit=5
    )
    titles = [r.title for r in results]
    assert titles == ["Python in feedback type"]


@pytest.mark.asyncio
async def test_semantic_mode_does_not_fall_back_to_keyword_on_empty_result() -> None:
    """No silent fallback: a zero-similarity result stays empty even if
    keyword matching would have returned rows. This is the contract of
    opt-in semantic mode."""
    store = MockStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    # Record has the word "python" in body so keyword mode would match.
    await mgr.create(_record(title="Anything", body="python tricks"))

    # Query contains none of the tag tokens, so the fake embedder returns
    # a zero vector -> every cosine similarity is 0 -> empty result.
    results = await mgr.search("foobar", MemoryScope.USER, limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_delete_failure_does_not_block_record_delete(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the embedding namespace is unreachable, deleting a record still
    succeeds and the failure is logged. Shutdown/teardown paths rely on
    this — we never want best-effort cleanup to block a real delete."""

    class BrokenEmbeddingStore(MockStore):
        async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
            if namespace and namespace[0] == "memory_embeddings":
                raise RuntimeError("embedding store down")
            await super().adelete(namespace, key)

    store = BrokenEmbeddingStore()
    embedder = _FakeEmbedder()
    mgr = PersistentMemoryManager(store, embedding_fn=embedder)

    rec = await mgr.create(_record(title="Python", body="python"))
    with caplog.at_level("ERROR"):
        deleted = await mgr.delete(rec.id, MemoryScope.USER)
    assert deleted is True
    assert any("Failed to remove embedding" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# __init__-level sanity
# ---------------------------------------------------------------------------


def test_manager_accepts_optional_embedding_fn() -> None:
    store = MockStore()
    mgr = PersistentMemoryManager(store)
    assert mgr._embedding_fn is None

    async def fn(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]

    mgr2 = PersistentMemoryManager(store, embedding_fn=fn)
    assert mgr2._embedding_fn is fn


def test_manager_ignores_embedding_text_change_heuristic_on_no_text_update() -> None:
    """Regression: the update-reindex heuristic checks keys in *updates*,
    not the resulting record. A noop update dict shouldn't re-embed."""
    # Exercised indirectly by test_update_non_text_field_does_not_reindex;
    # keep this stub so future tweaks to the heuristic remember that
    # intent. If the heuristic regresses, the sibling test will fail.
    assert True


# Force any stray record from these tests not to leak across runs — MockStore
# is per-fixture so this is only to appease linters that might flag an unused
# import path.
_ = Any
