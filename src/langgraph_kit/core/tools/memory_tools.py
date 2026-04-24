"""Agent-callable tools for managing persistent memory."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager


def build_memory_tools(memory_manager: PersistentMemoryManager) -> list[Any]:
    """Create tool functions for managing persistent memory.

    Returns a list of async callables suitable for passing to
    create_deep_agent(tools=...).
    """

    async def save_memory(
        title: str,
        memory_type: str,
        scope: str,
        summary: str,
        body: str,
    ) -> str:
        """Save a durable memory record for future conversations.

        Use this only for stable, future-useful facts that cannot be easily
        rediscovered from the current workspace. Do not save temporary task
        state, code patterns visible in the repo, or recent git history.

        Args:
            title: Short name for the memory
            memory_type: One of: user, feedback, project, reference
            scope: One of: user, assistant, project, team
            summary: One-line description used for retrieval
            body: Full memory content. For feedback type, include Why and How to apply.
        """
        try:
            mt = MemoryType(memory_type)
            ms = MemoryScope(scope)
        except ValueError:
            return (
                f"Error: invalid memory_type '{memory_type}' or scope '{scope}'. "
                f"Valid types: {[t.value for t in MemoryType]}. "
                f"Valid scopes: {[s.value for s in MemoryScope]}."
            )

        record = MemoryRecord(
            title=title,
            type=mt,
            scope=ms,
            summary=summary,
            body=body,
            source="agent_tool",
        )
        saved = await memory_manager.create(record)
        return f"Memory saved: [{saved.type.value}] {saved.title} (id: {saved.id})"

    async def list_memories(
        scope: str = "user",
        memory_type: str | None = None,
    ) -> str:
        """List stored memory records.

        Args:
            scope: One of: user, assistant, project, team
            memory_type: Optional filter. One of: user, feedback, project, reference
        """
        try:
            ms = MemoryScope(scope)
        except ValueError:
            return f"Error: invalid scope '{scope}'. Valid: {[s.value for s in MemoryScope]}."

        mt = None
        if memory_type is not None:
            try:
                mt = MemoryType(memory_type)
            except ValueError:
                return f"Error: invalid memory_type '{memory_type}'. Valid: {[t.value for t in MemoryType]}."

        records = await memory_manager.list_by_scope(ms, memory_type=mt, limit=20)
        if not records:
            return "No memories found."

        lines = [f"Found {len(records)} memories:\n"]
        for r in records:
            lines.append(f"- [{r.type.value}] {r.title}: {r.summary} (id: {r.id})")
        return "\n".join(lines)

    async def search_memories(
        query: str,
        scope: str = "user",
    ) -> str:
        """Search memories by semantic relevance.

        Args:
            query: Search query describing what you're looking for
            scope: One of: user, assistant, project, team
        """
        try:
            ms = MemoryScope(scope)
        except ValueError:
            return f"Error: invalid scope '{scope}'. Valid: {[s.value for s in MemoryScope]}."

        records = await memory_manager.search(query, ms, limit=5)
        if not records:
            return "No matching memories found."

        lines = [f"Found {len(records)} matching memories:\n"]
        for r in records:
            lines.append(
                f"- [{r.type.value}] {r.title}\n"
                + f"  Summary: {r.summary}\n"
                + f"  Body: {r.body}\n"
                + f"  (id: {r.id})"
            )
        return "\n".join(lines)

    async def update_memory(
        memory_id: str,
        scope: str,
        body: str,
        summary: str | None = None,
        memory_type: str | None = None,
        title: str | None = None,
    ) -> str:
        """Update an existing memory record.

        Args:
            memory_id: The ID of the memory to update
            scope: The scope where the memory lives (user, assistant, project, team)
            body: New body content
            summary: Optional new summary
            memory_type: Optional new type — re-classify a mis-typed record.
            title: Optional new title.
        """
        try:
            ms = MemoryScope(scope)
        except ValueError:
            return f"Error: invalid scope '{scope}'. Valid: {[s.value for s in MemoryScope]}."

        updates: dict[str, Any] = {"body": body}
        if summary is not None:
            updates["summary"] = summary
        if title is not None:
            updates["title"] = title
        if memory_type is not None:
            try:
                updates["type"] = MemoryType(memory_type)
            except ValueError:
                return (
                    f"Error: invalid memory_type '{memory_type}'. "
                    f"Valid: {[t.value for t in MemoryType]}."
                )

        result = await memory_manager.update(memory_id, ms, updates)
        if result is None:
            return f"Memory '{memory_id}' not found in scope '{scope}'."
        return f"Memory updated: [{result.type.value}] {result.title} (id: {result.id})"

    async def delete_memory(
        memory_id: str,
        scope: str,
    ) -> str:
        """Delete a memory record that is no longer relevant.

        Args:
            memory_id: The ID of the memory to delete
            scope: The scope where the memory lives (user, assistant, project, team)
        """
        try:
            ms = MemoryScope(scope)
        except ValueError:
            return f"Error: invalid scope '{scope}'. Valid: {[s.value for s in MemoryScope]}."

        deleted = await memory_manager.delete(memory_id, ms)
        if not deleted:
            return f"Memory '{memory_id}' not found in scope '{scope}'."
        return f"Memory '{memory_id}' deleted."

    return [save_memory, list_memories, search_memories, update_memory, delete_memory]
