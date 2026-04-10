"""Store-backed thread metadata management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThreadMetadata(BaseModel):
    """Metadata for a conversation thread."""

    thread_id: str
    user_id: str
    agent_id: str
    title: str = "New conversation"
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    last_message_preview: str = ""
    tags: list[str] = Field(default_factory=list)


class ThreadManager:
    """Store-backed thread metadata with user and agent indexing.

    Uses three namespace patterns:
    - Primary: ``("threads", thread_id)`` → full metadata
    - User index: ``("thread_index", "by_user", user_id)`` → keyed by thread_id
    - Agent index: ``("thread_index", "by_agent", agent_id)`` → keyed by thread_id
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    async def ensure_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_id: str,
        first_message: str | None = None,
    ) -> ThreadMetadata:
        """Create thread metadata if it doesn't exist, or update if it does."""
        existing = await self.get(thread_id)
        now = datetime.now(UTC).isoformat()

        if existing is not None:
            # Update existing thread
            updates: dict[str, Any] = {"updated_at": now}
            updates["message_count"] = existing.message_count + 1
            if first_message:
                updates["last_message_preview"] = first_message[:100]
            return await self._update_stored(existing, **updates)

        # Create new thread
        title = "New conversation"
        if first_message:
            title = first_message[:60].strip()
            if len(first_message) > 60:
                title += "..."

        meta = ThreadMetadata(
            thread_id=thread_id,
            user_id=user_id,
            agent_id=agent_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=1,
            last_message_preview=first_message[:100] if first_message else "",
        )
        await self._save(meta)
        return meta

    async def get(self, thread_id: str) -> ThreadMetadata | None:
        """Get thread metadata by ID."""
        try:
            items = await self._store.asearch(("threads", thread_id), limit=1)
            if items:
                return ThreadMetadata.model_validate(items[0].value)
        except Exception:
            logger.debug("Failed to get thread %s", thread_id, exc_info=True)
        return None

    async def list_for_user(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ThreadMetadata], int]:
        """List threads for a user, optionally filtered by agent.

        Returns (threads, total_count).
        """
        if agent_id:
            namespace = ("thread_index", "by_agent", agent_id)
        else:
            namespace = ("thread_index", "by_user", user_id)

        try:
            items = await self._store.asearch(namespace, limit=limit + offset + 100)

            # Filter to user's threads and collect thread IDs
            thread_refs = []
            for item in items:
                ref = item.value
                if ref.get("user_id") == user_id:
                    thread_refs.append(ref)

            # Sort by updated_at descending
            thread_refs.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
            total = len(thread_refs)

            # Apply pagination
            page = thread_refs[offset : offset + limit]

            # Load full metadata for the page
            threads: list[ThreadMetadata] = []
            for ref in page:
                tid = ref.get("thread_id", "")
                meta = await self.get(tid)
                if meta is not None:
                    threads.append(meta)

            return threads, total
        except Exception:
            logger.debug("Failed to list threads for user %s", user_id, exc_info=True)
            return [], 0

    async def update(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> ThreadMetadata | None:
        """Update thread metadata fields."""
        existing = await self.get(thread_id)
        if existing is None:
            return None

        updates: dict[str, Any] = {"updated_at": datetime.now(UTC).isoformat()}
        if title is not None:
            updates["title"] = title
        if tags is not None:
            updates["tags"] = tags
        return await self._update_stored(existing, **updates)

    async def delete(self, thread_id: str) -> bool:
        """Delete thread metadata and index entries."""
        existing = await self.get(thread_id)
        if existing is None:
            return False

        try:
            await self._store.adelete(("threads", thread_id), "metadata")
            await self._store.adelete(
                ("thread_index", "by_user", existing.user_id), thread_id
            )
            await self._store.adelete(
                ("thread_index", "by_agent", existing.agent_id), thread_id
            )
            return True
        except Exception:
            logger.debug("Failed to delete thread %s", thread_id, exc_info=True)
            return False

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 20,
    ) -> list[ThreadMetadata]:
        """Simple keyword search on title and last_message_preview."""
        threads, _ = await self.list_for_user(user_id, limit=200)
        query_lower = query.lower()
        results = [
            t
            for t in threads
            if query_lower in t.title.lower()
            or query_lower in t.last_message_preview.lower()
        ]
        return results[:limit]

    async def _save(self, meta: ThreadMetadata) -> None:
        """Save thread metadata and index entries."""
        data = meta.model_dump(mode="json")
        await self._store.aput(("threads", meta.thread_id), "metadata", data)

        # Index entries (minimal data for listing)
        ref = {
            "thread_id": meta.thread_id,
            "user_id": meta.user_id,
            "updated_at": meta.updated_at,
        }
        await self._store.aput(
            ("thread_index", "by_user", meta.user_id), meta.thread_id, ref
        )
        await self._store.aput(
            ("thread_index", "by_agent", meta.agent_id), meta.thread_id, ref
        )

    async def _update_stored(
        self, existing: ThreadMetadata, **updates: Any
    ) -> ThreadMetadata:
        """Apply updates to existing metadata and save."""
        data = existing.model_dump(mode="json")
        data.update(updates)
        updated = ThreadMetadata.model_validate(data)
        await self._save(updated)
        return updated
