"""Background memory consolidation — merges duplicates, prunes stale entries, normalizes records."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.memory._parsing import parse_json_array
from langgraph_kit.core.memory.models import MemoryRecord, MemoryScope, MemoryType
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

logger = logging.getLogger(__name__)

_CONSOLIDATION_PROMPT = """You are a memory maintenance worker. Review the following memory records and identify improvements.

## Rules
- MERGE near-duplicates into a single, better record
- DELETE records that are stale, no longer relevant, or derivable from the current environment
- UPDATE records that need correction or clarification
- KEEP records that are accurate and useful as-is
- Be CONSERVATIVE — when in doubt, keep the record
- Never invent new facts — only reorganize existing ones

## Current Memories
{memories}

## Output Format
Return a JSON array of actions. Each action is one of:
- {{"action": "keep", "id": "..."}}
- {{"action": "delete", "id": "...", "reason": "..."}}
- {{"action": "merge", "source_ids": ["id1", "id2"], "merged": {{"title": "...", "type": "...", "summary": "...", "body": "..."}}}}
- {{"action": "update", "id": "...", "updates": {{"body": "...", "summary": "..."}}}}

Respond with ONLY the JSON array.
"""


class ConsolidationResult:
    """Tracks what happened during a consolidation pass."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: int = 0
        self.deleted: int = 0
        self.merged: int = 0
        self.updated: int = 0
        self.errors: list[str] = []

    @property
    def total_actions(self) -> int:
        return self.kept + self.deleted + self.merged + self.updated

    def __repr__(self) -> str:
        return (
            f"ConsolidationResult(kept={self.kept}, deleted={self.deleted}, "
            f"merged={self.merged}, updated={self.updated}, errors={len(self.errors)})"
        )


class MemoryConsolidator:
    """Runs a consolidation pass over stored memories using an LLM."""

    def __init__(self, memory_manager: PersistentMemoryManager, llm: Any) -> None:
        super().__init__()
        self._memory = memory_manager
        self._llm = llm

    async def consolidate(
        self,
        scope: MemoryScope = MemoryScope.USER,
        limit: int = 100,
    ) -> ConsolidationResult:
        """Run a consolidation pass on memories in the given scope.

        1. Loads all memories in scope
        2. Asks LLM to identify merges, deletes, updates
        3. Applies the actions
        4. Returns a result summary
        """
        result = ConsolidationResult()

        records = await self._memory.list_by_scope(scope, limit=limit)
        if len(records) < 2:
            # Nothing to consolidate
            return result

        # Build prompt
        memories_text = self._format_records(records)
        prompt = _CONSOLIDATION_PROMPT.format(memories=memories_text)

        # Call LLM
        try:
            from langchain_core.messages import HumanMessage

            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception:
            logger.exception("Consolidation LLM call failed")
            result.errors.append("LLM call failed")
            return result

        # Parse and apply actions
        actions = self._parse_response(raw)
        await self._apply_actions(actions, scope, result)

        logger.info("Consolidation complete: %s", result)
        return result

    def _format_records(self, records: list[MemoryRecord]) -> str:
        lines: list[str] = []
        for r in records:
            lines.append(
                f"- id: {r.id}\n"
                + f"  type: {r.type.value}\n"
                + f"  title: {r.title}\n"
                + f"  summary: {r.summary}\n"
                + f"  body: {r.body}\n"
                + f"  updated: {r.updated_at.isoformat()}"
            )
        return "\n\n".join(lines)

    def _parse_response(self, raw: str) -> list[dict[str, Any]]:
        return parse_json_array(raw, context="consolidation response")

    async def _apply_actions(
        self,
        actions: list[dict[str, Any]],
        scope: MemoryScope,
        result: ConsolidationResult,
    ) -> None:
        for action in actions:
            try:
                act = action.get("action", "keep")

                if act == "keep":
                    result.kept += 1

                elif act == "delete":
                    record_id = action.get("id", "")
                    if record_id:
                        deleted = await self._memory.delete(record_id, scope)
                        if deleted:
                            result.deleted += 1
                            logger.info(
                                "Consolidated: deleted %s — %s",
                                record_id,
                                action.get("reason", "no reason"),
                            )

                elif act == "merge":
                    source_ids = action.get("source_ids", [])
                    merged_data = action.get("merged", {})
                    if source_ids and merged_data:
                        # Delete source records
                        for sid in source_ids:
                            await self._memory.delete(sid, scope)
                        # Create merged record
                        record = MemoryRecord(
                            title=merged_data.get("title", "Merged"),
                            type=MemoryType(merged_data.get("type", "user")),
                            scope=scope,
                            summary=merged_data.get("summary", ""),
                            body=merged_data.get("body", ""),
                            source="consolidation_merge",
                        )
                        await self._memory.create(record)
                        result.merged += 1
                        logger.info(
                            "Consolidated: merged %s into %s", source_ids, record.id
                        )

                elif act == "update":
                    record_id = action.get("id", "")
                    updates = action.get("updates", {})
                    if record_id and updates:
                        updated = await self._memory.update(record_id, scope, updates)
                        if updated:
                            result.updated += 1
                            logger.info("Consolidated: updated %s", record_id)

            except Exception:
                msg = f"Failed to apply action: {action}"
                logger.exception(msg)
                result.errors.append(msg)
