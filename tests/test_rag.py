"""Coverage — RAG ingestion, RetrievalIndex round trip, and search_knowledge tool.

Issue #16 lands the foundation: chunker, vector index built on a
caller-supplied embedding fn, and the agent-facing
``search_knowledge`` tool. Citation verification + the grounding
eval rubric live in a follow-up.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Document,
    RetrievalIndex,
    build_search_knowledge_tool,
    word_chunker,
)
from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Fake embedder — same trick used in test_memory_semantic_search
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """3-dim vector keyed by counts of {dog, car, python} substrings."""

    TAGS = ("dog", "car", "python")

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(t.lower().count(tag)) for tag in self.TAGS] for t in texts]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def test_word_chunker_returns_full_text_when_under_size() -> None:
    text = "Hello world."
    assert word_chunker(text, chunk_size=100, chunk_overlap=10) == [text]


def test_word_chunker_splits_at_word_boundaries() -> None:
    # 80 words of "alpha beta " -> ~880 chars; chunk into 200-char windows.
    text = ("alpha beta " * 80).strip()
    chunks = word_chunker(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # No chunk should start or end mid-word.
        assert not c.startswith(" ")
        assert not c.endswith(" ")
        # No chunk should split a word — every word in the result is a
        # whole word from {alpha, beta}.
        for word in c.split():
            assert word in {"alpha", "beta"}


def test_word_chunker_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        word_chunker("text", chunk_size=10, chunk_overlap=10)


def test_default_chunk_size_constants_are_reasonable() -> None:
    # Sanity-check the public defaults — ~3.2k chars per chunk with 200
    # overlap is the contract documented for the RAG module.
    assert DEFAULT_CHUNK_SIZE == 3200
    assert DEFAULT_CHUNK_OVERLAP == 200


# ---------------------------------------------------------------------------
# RetrievalIndex round trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_and_search_round_trip() -> None:
    store = MockStore()
    index = RetrievalIndex(
        "docs", store, _FakeEmbedder(), chunk_size=200, chunk_overlap=20
    )

    await index.aput_document(Document(id="d1", text="all about python python"))
    await index.aput_document(Document(id="d2", text="cars and more cars and cars"))
    await index.aput_document(Document(id="d3", text="dogs everywhere"))

    results = await index.asearch("python guide", top_k=3)
    assert results, "expected at least one result"
    # The python doc should rank first under cosine on the python axis.
    assert results[0].doc_id == "d1"
    # Any returned chunk should carry score > 0 (drop_zero filter).
    for r in results:
        assert r.score > 0.0


@pytest.mark.asyncio
async def test_ingest_replaces_chunks_for_same_doc_id() -> None:
    store = MockStore()
    index = RetrievalIndex(
        "docs", store, _FakeEmbedder(), chunk_size=80, chunk_overlap=10
    )

    await index.aput_document(Document(id="d1", text="python python python"))
    initial = await index.asearch("python", top_k=10)
    assert len(initial) >= 1

    # Re-ingest with new content — old chunks should be replaced.
    await index.aput_document(Document(id="d1", text="cars cars cars"))
    refreshed = await index.asearch("python", top_k=10)
    # Now the doc has no python tokens, so semantic search returns nothing.
    assert refreshed == []
    # ...but car queries should hit it.
    car_results = await index.asearch("cars", top_k=10)
    assert any(r.doc_id == "d1" for r in car_results)


@pytest.mark.asyncio
async def test_delete_document_removes_chunks_and_embeddings() -> None:
    store = MockStore()
    index = RetrievalIndex(
        "docs", store, _FakeEmbedder(), chunk_size=80, chunk_overlap=10
    )

    await index.aput_document(Document(id="d1", text="python python python"))
    deleted = await index.adelete_document("d1")
    assert deleted >= 1
    assert await index.asearch("python", top_k=5) == []

    # Underlying namespaces should be empty for d1.
    chunk_ns = ("rag", "docs", "chunks")
    embedding_ns = ("rag", "docs", "embeddings")
    chunk_keys = [k for k in store._data.get(chunk_ns, {}) if k.startswith("d1:")]
    embedding_keys = [
        k for k in store._data.get(embedding_ns, {}) if k.startswith("d1:")
    ]
    assert chunk_keys == []
    assert embedding_keys == []


@pytest.mark.asyncio
async def test_delete_unknown_doc_id_is_noop() -> None:
    store = MockStore()
    index = RetrievalIndex("docs", store, _FakeEmbedder())
    assert await index.adelete_document("nope") == 0


@pytest.mark.asyncio
async def test_search_with_empty_query_returns_empty() -> None:
    store = MockStore()
    index = RetrievalIndex("docs", store, _FakeEmbedder())
    await index.aput_document(Document(id="d1", text="anything"))
    assert await index.asearch("   ", top_k=5) == []


@pytest.mark.asyncio
async def test_search_with_zero_top_k_returns_empty() -> None:
    store = MockStore()
    index = RetrievalIndex("docs", store, _FakeEmbedder())
    await index.aput_document(Document(id="d1", text="python"))
    assert await index.asearch("python", top_k=0) == []


@pytest.mark.asyncio
async def test_embedding_fn_returning_wrong_count_raises() -> None:
    class _BrokenEmbedder:
        async def __call__(self, texts: list[str]) -> list[list[float]]:
            # Always returns one vector regardless of input count
            _ = texts
            return [[0.0, 0.0, 0.0]]

    store = MockStore()
    # Force chunk_size=10 so a longer text yields multiple chunks.
    index = RetrievalIndex(
        "docs", store, _BrokenEmbedder(), chunk_size=10, chunk_overlap=2
    )
    with pytest.raises(ValueError, match="vectors"):
        await index.aput_document(Document(id="d1", text="word " * 50))


# ---------------------------------------------------------------------------
# search_knowledge tool factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_tool_returns_sentinel_on_empty() -> None:
    store = MockStore()
    index = RetrievalIndex("docs", store, _FakeEmbedder())
    tool = build_search_knowledge_tool(index)

    result = await tool.fn(query="anything", top_k=3)
    assert "no matching passages" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_tool_formats_ranked_results() -> None:
    store = MockStore()
    index = RetrievalIndex(
        "docs", store, _FakeEmbedder(), chunk_size=200, chunk_overlap=20
    )
    await index.aput_document(Document(id="doc-a", text="all about python python"))
    await index.aput_document(Document(id="doc-b", text="cars cars"))
    tool = build_search_knowledge_tool(index, default_top_k=2)

    result = await tool.fn(query="python guide")
    assert "Found" in result
    assert "doc-a" in result
    # The chunk preview should be present (truncated if huge).
    assert "python" in result.lower()


def test_search_knowledge_tool_capability_metadata() -> None:
    store = MockStore()
    index = RetrievalIndex("docs", store, _FakeEmbedder())
    tool = build_search_knowledge_tool(index)
    assert tool.id == "rag.search_knowledge"
    assert tool.name == "search_knowledge"
    assert "rag" in tool.tags
    assert tool.prompt_guidance is not None


# ---------------------------------------------------------------------------
# top_k_by_score helper (sanity)
# ---------------------------------------------------------------------------


def test_top_k_by_score_drops_zero_and_sorts_descending() -> None:
    from langgraph_kit.core._vector_math import top_k_by_score

    items: list[tuple[float, Any]] = [(0.0, "a"), (0.5, "b"), (0.9, "c"), (0.0, "d")]
    out = top_k_by_score(items, k=2)
    assert out == [(0.9, "c"), (0.5, "b")]
