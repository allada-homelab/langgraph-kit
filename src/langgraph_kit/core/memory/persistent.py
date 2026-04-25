"""Persistent memory manager providing CRUD operations on top of LangGraph Store."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from langgraph_kit.core._vector_math import cosine_similarity
from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)

logger = logging.getLogger(__name__)

_ALL_MEMORY_TYPES: list[MemoryType] = list(MemoryType)

# Parallel namespace for embedding vectors, keyed by record id. Separate
# from the main record namespace so the Store can still return records
# without embedding bloat, and so we can delete embeddings independently.
_EMBEDDING_NS_PREFIX = "memory_embeddings"

# Type alias for the batch embedding function signature.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _text_for_embedding(record: MemoryRecord) -> str:
    """Concatenate the searchable text of a record for embedding."""
    # Title carries the highest signal; weight it implicitly by ordering.
    return f"{record.title}\n{record.summary}\n{record.body}"


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    """Case-insensitive word tokens. Keyword fallback uses set overlap."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


_cosine = cosine_similarity


class PersistentMemoryManager:
    """CRUD facade over a LangGraph BaseStore for typed memory records.

    Optional semantic search: pass ``embedding_fn`` (or set
    ``AgentConfig.memory_embedding_fn``) to enable vector-similarity search.
    When no embedding function is configured, :meth:`search` uses
    case-insensitive token overlap against title/summary/body. The presence
    of the callable is the *only* switch — there is no silent fallback from
    a failed semantic query to keyword, because that would make behavior
    depend on which store happened to implement ``asearch(query=...)``.
    """

    def __init__(
        self,
        store: Any,
        namespace_prefix: str = "memory",
        *,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._prefix = namespace_prefix
        self._embedding_fn: EmbeddingFn | None = embedding_fn

    # ------------------------------------------------------------------
    # Namespaces
    # ------------------------------------------------------------------

    def _namespace(
        self,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
    ) -> tuple[str, ...]:
        """Build a store namespace tuple.

        Examples:
            ("memory", "user", "feedback")
            ("memory", "user")  — when memory_type is None
        """
        base = (self._prefix, scope.value)
        if memory_type is not None:
            return (*base, memory_type.value)
        return base

    def _embedding_namespace(self, scope: MemoryScope) -> tuple[str, ...]:
        """Single namespace per scope for embedding vectors."""
        return (_EMBEDDING_NS_PREFIX, scope.value)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _index_record(self, record: MemoryRecord) -> None:
        """Compute and persist the embedding vector for a record."""
        if self._embedding_fn is None:
            return
        vectors = await self._embedding_fn([_text_for_embedding(record)])
        if not vectors:
            return
        await self._store.aput(
            self._embedding_namespace(record.scope),
            record.id,
            {
                "vector": list(vectors[0]),
                "memory_type": record.type.value,
                "updated_at": record.updated_at.isoformat(),
            },
        )

    async def _unindex_record(self, record_id: str, scope: MemoryScope) -> None:
        """Best-effort embedding removal — failure is logged but non-fatal."""
        if self._embedding_fn is None:
            return
        try:
            await self._store.adelete(self._embedding_namespace(scope), record_id)
        except Exception:
            logger.exception(
                "Failed to remove embedding for record %s in scope %s",
                record_id,
                scope.value,
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record and return it."""
        ns = self._namespace(record.scope, record.type)
        await self._store.aput(ns, record.id, record.to_store_value())
        await self._index_record(record)
        return record

    async def get(
        self,
        record_id: str,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
    ) -> MemoryRecord | None:
        """Retrieve a record by id, self-healing duplicate-namespace state.

        If memory_type is given, looks up directly (1 round-trip). Otherwise
        searches across all type namespaces. When a record ID exists under
        more than one type namespace — which can happen if an earlier
        ``update()`` across type changes crashed between the write-new and
        delete-old steps — the most-recently-updated record wins and the
        stale copies are deleted opportunistically so subsequent reads are
        consistent.
        """
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        hits: list[tuple[tuple[str, ...], MemoryRecord]] = []
        for mt in types:
            ns = self._namespace(scope, mt)
            item = await self._store.aget(ns, record_id)
            if item is not None:
                hits.append((ns, MemoryRecord.from_store_value(item.value)))

        if not hits:
            return None
        if len(hits) == 1:
            return hits[0][1]

        # Duplicate — reconcile. Pick the newest by updated_at; drop the
        # rest to prevent the inconsistency from propagating.
        hits.sort(key=lambda pair: pair[1].updated_at, reverse=True)
        newest_ns, newest = hits[0]
        for stale_ns, stale in hits[1:]:
            logger.warning(
                "Cleaning up orphan memory record %s in namespace %r (winner in %r)",
                record_id,
                stale_ns,
                newest_ns,
            )
            try:
                await self._store.adelete(stale_ns, record_id)
            except Exception:
                logger.exception(
                    "Failed to delete orphan memory %s at %r", record_id, stale_ns
                )
            _ = stale  # silence unused-tracking
        return newest

    async def update(
        self,
        record_id: str,
        scope: MemoryScope,
        updates: dict[str, Any],
        memory_type: MemoryType | None = None,
    ) -> MemoryRecord | None:
        """Apply partial updates to an existing record. Returns None if not found."""
        existing = await self.get(record_id, scope, memory_type)
        if existing is None:
            return None

        merged = existing.model_dump()
        merged.update(updates)
        merged["updated_at"] = datetime.now(UTC)

        updated = MemoryRecord.model_validate(merged)
        ns = self._namespace(updated.scope, updated.type)
        await self._store.aput(ns, updated.id, updated.to_store_value())

        # If the type or scope changed, remove the record from the old namespace.
        if existing.type != updated.type or existing.scope != updated.scope:
            old_ns = self._namespace(existing.scope, existing.type)
            await self._store.adelete(old_ns, record_id)

        # Re-index if embedding-relevant fields changed, or if scope changed
        # (embeddings are namespaced by scope, so a scope change needs a move).
        embedding_text_changed = any(k in updates for k in ("title", "summary", "body"))
        if existing.scope != updated.scope:
            # Move the embedding to the new scope's namespace.
            await self._unindex_record(record_id, existing.scope)
            await self._index_record(updated)
        elif embedding_text_changed:
            await self._index_record(updated)

        return updated

    async def delete(
        self,
        record_id: str,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a record by id. Returns True if found and deleted."""
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        for mt in types:
            ns = self._namespace(scope, mt)
            item = await self._store.aget(ns, record_id)
            if item is not None:
                await self._store.adelete(ns, record_id)
                await self._unindex_record(record_id, scope)
                return True
        return False

    async def list_by_scope(
        self,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List records in a scope, optionally filtered by type."""
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        records: list[MemoryRecord] = []
        remaining = limit

        for mt in types:
            if remaining <= 0:
                break
            ns = self._namespace(scope, mt)
            items: list[Any] = await self._store.asearch(ns, limit=remaining)
            for item in items:
                records.append(MemoryRecord.from_store_value(item.value))
            remaining = limit - len(records)

        return records

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Return records most relevant to *query* within *scope*.

        Ranking mode is chosen once by whether an embedding function was
        supplied at construction (or via ``AgentConfig.memory_embedding_fn``):

        - **Embedding-backed**: query is embedded, candidate vectors in the
          scope's embedding namespace are ranked by cosine similarity, and
          the top-K matching records are fetched.
        - **Keyword**: records in the scope are scored by case-insensitive
          token overlap with the query against title + summary + body.

        No silent fallback — the caller's embedding-function choice
        determines the mode deterministically.
        """
        if self._embedding_fn is not None:
            return await self._semantic_search(query, scope, memory_type, limit)
        return await self._keyword_search(query, scope, memory_type, limit)

    async def _semantic_search(
        self,
        query: str,
        scope: MemoryScope,
        memory_type: MemoryType | None,
        limit: int,
    ) -> list[MemoryRecord]:
        """Rank by cosine similarity on embeddings. Assumes embedding_fn is set."""
        embedding_fn = self._embedding_fn
        if embedding_fn is None:
            return []
        q_vectors = await embedding_fn([query])
        if not q_vectors:
            return []
        q_vec = list(q_vectors[0])

        ns = self._embedding_namespace(scope)
        items: list[Any] = await self._store.asearch(ns, limit=1000)

        scored: list[tuple[float, str, str]] = []  # (sim, record_id, type)
        for item in items:
            payload = item.value
            if (
                memory_type is not None
                and payload.get("memory_type") != memory_type.value
            ):
                continue
            vec = payload.get("vector") or []
            sim = _cosine(q_vec, vec)
            scored.append((sim, item.key, payload.get("memory_type", "")))
        scored.sort(key=lambda t: t[0], reverse=True)

        records: list[MemoryRecord] = []
        for sim, rec_id, type_str in scored[:limit]:
            if sim <= 0.0:
                continue
            mt = MemoryType(type_str) if type_str else None
            record = await self.get(rec_id, scope, mt)
            if record is not None:
                records.append(record)
        return records

    async def _keyword_search(
        self,
        query: str,
        scope: MemoryScope,
        memory_type: MemoryType | None,
        limit: int,
    ) -> list[MemoryRecord]:
        """Rank by query-token overlap against title+summary+body."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Pull a broader candidate set so we can rank and trim to limit.
        candidates = await self.list_by_scope(scope, memory_type, limit=1000)

        scored: list[tuple[int, MemoryRecord]] = []
        for rec in candidates:
            text = f"{rec.title}\n{rec.summary}\n{rec.body}"
            overlap = len(_tokenize(text) & q_tokens)
            if overlap > 0:
                scored.append((overlap, rec))
        # Sort by descending overlap, then by recency as a tiebreaker.
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return [rec for _, rec in scored[:limit]]

    async def list_all_scopes(self) -> list[MemoryScope]:
        """Return scopes that contain at least one record."""
        found: list[MemoryScope] = []
        for scope in MemoryScope:
            ns = self._namespace(scope)
            namespaces: list[Any] = await self._store.alist_namespaces(prefix=ns)
            if namespaces:
                found.append(scope)
        return found
