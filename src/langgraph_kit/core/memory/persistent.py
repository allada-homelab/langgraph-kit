"""Persistent memory manager providing CRUD operations on top of LangGraph Store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)

_ALL_MEMORY_TYPES: list[MemoryType] = list(MemoryType)


class PersistentMemoryManager:
    """CRUD facade over a LangGraph BaseStore for typed memory records."""

    def __init__(self, store: Any, namespace_prefix: str = "memory") -> None:
        super().__init__()
        self._store = store
        self._prefix = namespace_prefix

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

    async def create(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record and return it."""
        ns = self._namespace(record.scope, record.type)
        await self._store.aput(ns, record.id, record.to_store_value())
        return record

    async def get(
        self,
        record_id: str,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
    ) -> MemoryRecord | None:
        """Retrieve a record by id.

        If memory_type is given, looks up directly (1 round-trip).
        Otherwise searches across all type namespaces (up to 4 round-trips).
        """
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        for mt in types:
            ns = self._namespace(scope, mt)
            item = await self._store.aget(ns, record_id)
            if item is not None:
                return MemoryRecord.from_store_value(item.value)
        return None

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
        merged["updated_at"] = datetime.now(UTC).isoformat()

        updated = MemoryRecord.model_validate(merged)
        ns = self._namespace(updated.scope, updated.type)
        await self._store.aput(ns, updated.id, updated.to_store_value())

        # If the type changed, remove the record from the old namespace.
        if existing.type != updated.type:
            old_ns = self._namespace(existing.scope, existing.type)
            await self._store.adelete(old_ns, record_id)

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

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Semantic search for records matching a query string."""
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        records: list[MemoryRecord] = []
        remaining = limit

        for mt in types:
            if remaining <= 0:
                break
            ns = self._namespace(scope, mt)
            items: list[Any] = await self._store.asearch(
                ns, query=query, limit=remaining
            )
            for item in items:
                records.append(MemoryRecord.from_store_value(item.value))
            remaining = limit - len(records)

        return records

    async def list_all_scopes(self) -> list[MemoryScope]:
        """Return scopes that contain at least one record."""
        found: list[MemoryScope] = []
        for scope in MemoryScope:
            ns = self._namespace(scope)
            namespaces: list[Any] = await self._store.alist_namespaces(prefix=ns)
            if namespaces:
                found.append(scope)
        return found
