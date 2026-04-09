"""Worker-scoped memory for specialized agents to retain role-specific knowledge."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.memory.models import MemoryRecord, MemoryType

_ALL_MEMORY_TYPES: list[MemoryType] = list(MemoryType)

logger = logging.getLogger(__name__)


class AgentMemoryManager:
    """Memory manager scoped to a specific worker/agent role.

    Stores memories under namespace ("memory", "agent", agent_name, type).
    This keeps worker-specific knowledge separate from user/project/team memory.

    Example: A "researcher" agent might remember:
    - Preferred search patterns for this codebase
    - Known architecture conventions
    - Common false-positive patterns to skip
    """

    def __init__(
        self, store: Any, agent_name: str, namespace_prefix: str = "memory"
    ) -> None:
        super().__init__()
        self._store = store
        self._agent_name = agent_name
        self._prefix = namespace_prefix

    def _namespace(self, memory_type: MemoryType | None = None) -> tuple[str, ...]:
        base = (self._prefix, "agent", self._agent_name)
        if memory_type is not None:
            return (*base, memory_type.value)
        return base

    async def create(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a worker-scoped memory record."""
        ns = self._namespace(record.type)
        await self._store.aput(ns, record.id, record.to_store_value())
        return record

    async def get(
        self, record_id: str, memory_type: MemoryType | None = None
    ) -> MemoryRecord | None:
        """Retrieve a record by id.

        If memory_type is given, looks up directly (1 round-trip).
        Otherwise searches across all type namespaces.
        """
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        for mt in types:
            ns = self._namespace(mt)
            item = await self._store.aget(ns, record_id)
            if item is not None:
                return MemoryRecord.from_store_value(item.value)
        return None

    async def list_all(
        self, memory_type: MemoryType | None = None, limit: int = 50
    ) -> list[MemoryRecord]:
        """List all memories for this agent, optionally filtered by type."""
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        records: list[MemoryRecord] = []
        remaining = limit
        for mt in types:
            if remaining <= 0:
                break
            ns = self._namespace(mt)
            items: list[Any] = await self._store.asearch(ns, limit=remaining)
            for item in items:
                records.append(MemoryRecord.from_store_value(item.value))
            remaining = limit - len(records)
        return records

    async def update(
        self,
        record_id: str,
        updates: dict[str, Any],
        memory_type: MemoryType | None = None,
    ) -> MemoryRecord | None:
        """Apply partial updates to an existing record. Returns None if not found."""
        from datetime import UTC, datetime

        existing = await self.get(record_id, memory_type)
        if existing is None:
            return None

        merged = existing.model_dump()
        merged.update(updates)
        merged["updated_at"] = datetime.now(UTC).isoformat()

        updated = MemoryRecord.model_validate(merged)
        ns = self._namespace(updated.type)
        await self._store.aput(ns, updated.id, updated.to_store_value())

        # If the type changed, remove the record from the old namespace.
        if existing.type != updated.type:
            old_ns = self._namespace(existing.type)
            await self._store.adelete(old_ns, record_id)

        return updated

    async def delete(
        self, record_id: str, memory_type: MemoryType | None = None
    ) -> bool:
        """Delete a record by id. Returns True if found and deleted."""
        types = [memory_type] if memory_type is not None else _ALL_MEMORY_TYPES
        for mt in types:
            ns = self._namespace(mt)
            item = await self._store.aget(ns, record_id)
            if item is not None:
                await self._store.adelete(ns, record_id)
                return True
        return False

    async def snapshot_from(self, source_records: list[MemoryRecord]) -> int:
        """Initialize agent memory from a list of source records.

        Useful for seeding a worker with relevant user or project memories
        at the start of a task. Returns the number of records written.
        """
        count = 0
        for record in source_records:
            agent_record = MemoryRecord(
                title=record.title,
                type=record.type,
                scope=record.scope,
                summary=record.summary,
                body=record.body,
                source=f"snapshot_from:{record.id}",
            )
            await self.create(agent_record)
            count += 1
        return count
