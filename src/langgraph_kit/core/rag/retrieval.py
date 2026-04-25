"""Vector-similarity retrieval over a LangGraph ``BaseStore``.

The :class:`RetrievalIndex` writes chunks and embeddings to two
parallel namespaces and ranks queries by cosine similarity. Storage
contract:

- ``("rag", index_name, "chunks")`` keyed by ``chunk_id`` →
  ``{"text": str, "doc_id": str, "metadata": dict, "position": int}``
- ``("rag", index_name, "embeddings")`` keyed by ``chunk_id`` →
  ``{"vector": list[float], "doc_id": str}``
- ``("rag", index_name, "doc_chunks")`` keyed by ``doc_id`` →
  ``{"chunk_ids": list[str]}`` — used for fast delete/replace.

Splitting chunks and embeddings keeps the chunk payload light when
read by tools (no ~6 KB vector tax on every retrieval) and lets the
vector index be rebuilt without reingesting source text.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from langgraph_kit.core._vector_math import cosine_similarity, top_k_by_score

from .ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunker,
    Document,
    word_chunker,
)

logger = logging.getLogger(__name__)

# Same shape as the memory module's EmbeddingFn so a single embedder
# can back both. Kept as a re-imported type alias rather than imported
# from memory to keep RAG / memory decoupled — they share an *interface*,
# not an implementation.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _chunk_namespace(index_name: str) -> tuple[str, ...]:
    return ("rag", index_name, "chunks")


def _embedding_namespace(index_name: str) -> tuple[str, ...]:
    return ("rag", index_name, "embeddings")


def _doc_chunks_namespace(index_name: str) -> tuple[str, ...]:
    return ("rag", index_name, "doc_chunks")


class RetrievedChunk(BaseModel):
    """A chunk returned by :meth:`RetrievalIndex.asearch`."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    position: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalIndex:
    """Document index backed by a LangGraph Store and a caller-supplied embedder.

    The chunker and embedding function are explicit — no provider lock
    in. Re-ingesting the same ``doc_id`` replaces all of its chunks,
    so updates are write-through. Deletes are scoped to a single
    document; per-namespace clears require iterating doc_chunks.
    """

    def __init__(
        self,
        name: str,
        store: Any,
        embedding_fn: EmbeddingFn,
        *,
        chunker: Chunker | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        super().__init__()
        self._name = name
        self._store = store
        self._embedding_fn = embedding_fn
        self._chunker = chunker or (
            lambda text: word_chunker(
                text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    async def aput_document(self, document: Document) -> int:
        """Ingest *document*, returning the number of chunks created.

        If the document id already exists in the index, its previous
        chunks (and embeddings) are removed first so the call is idempotent.
        """
        await self.adelete_document(document.id)

        chunks = self._chunker(document.text)
        if not chunks:
            return 0

        vectors = await self._embedding_fn(chunks)
        if len(vectors) != len(chunks):
            msg = (
                f"embedding_fn returned {len(vectors)} vectors for "
                f"{len(chunks)} chunks; expected one per chunk"
            )
            raise ValueError(msg)

        chunk_ids: list[str] = []
        for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            chunk_id = f"{document.id}:{position}"
            chunk_ids.append(chunk_id)
            await self._store.aput(
                _chunk_namespace(self._name),
                chunk_id,
                {
                    "text": chunk,
                    "doc_id": document.id,
                    "position": position,
                    "metadata": dict(document.metadata),
                },
            )
            await self._store.aput(
                _embedding_namespace(self._name),
                chunk_id,
                {"vector": list(vector), "doc_id": document.id},
            )

        await self._store.aput(
            _doc_chunks_namespace(self._name),
            document.id,
            {"chunk_ids": chunk_ids},
        )
        return len(chunk_ids)

    async def adelete_document(self, doc_id: str) -> int:
        """Remove *doc_id* and all its chunks/embeddings. Returns count deleted."""
        item = await self._store.aget(_doc_chunks_namespace(self._name), doc_id)
        if item is None:
            return 0
        chunk_ids = list((item.value or {}).get("chunk_ids", []))
        for chunk_id in chunk_ids:
            try:
                await self._store.adelete(_chunk_namespace(self._name), chunk_id)
            except Exception:
                logger.exception(
                    "Failed to delete chunk %s for doc %s", chunk_id, doc_id
                )
            try:
                await self._store.adelete(_embedding_namespace(self._name), chunk_id)
            except Exception:
                logger.exception(
                    "Failed to delete embedding %s for doc %s", chunk_id, doc_id
                )
        try:
            await self._store.adelete(_doc_chunks_namespace(self._name), doc_id)
        except Exception:
            logger.exception("Failed to delete doc_chunks index entry %s", doc_id)
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def asearch(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top-K chunks closest to *query* by cosine similarity."""
        if not query.strip() or top_k <= 0:
            return []

        q_vectors = await self._embedding_fn([query])
        if not q_vectors:
            return []
        q_vec = list(q_vectors[0])

        embedding_items: list[Any] = await self._store.asearch(
            _embedding_namespace(self._name), limit=10_000
        )

        scored: list[tuple[float, str]] = []
        for item in embedding_items:
            vec = (item.value or {}).get("vector") or []
            scored.append((cosine_similarity(q_vec, vec), item.key))

        ranked = top_k_by_score(scored, k=top_k, drop_zero=True)

        results: list[RetrievedChunk] = []
        for score, chunk_id_obj in ranked:  # pyright: ignore[reportGeneralTypeIssues]
            if not isinstance(chunk_id_obj, str):
                # top_k_by_score is generic on payload type; we only
                # ever push str keys, so this is a defensive narrow
                # rather than a real branch.
                continue
            chunk_item = await self._store.aget(
                _chunk_namespace(self._name), chunk_id_obj
            )
            if chunk_item is None:
                # Embedding without a backing chunk — orphaned. Skip and
                # let the next ingest cycle clean up; not fatal for the
                # current query.
                continue
            payload = chunk_item.value or {}
            chunk_id = chunk_id_obj
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", ""),
                    score=float(score),
                    position=int(payload.get("position", 0)),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return results
