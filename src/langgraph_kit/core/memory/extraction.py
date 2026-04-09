"""Post-turn automatic memory extraction from recent conversation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

logger = logging.getLogger(__name__)

_RE_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_MAX_MESSAGE_CHARS = 2000  # truncate messages longer than this when formatting for LLM

_EXTRACTION_PROMPT = """You are a memory extraction worker. Your ONLY job is to identify durable, future-useful facts from the recent conversation that should be saved to long-term memory.

## Rules
- Save ONLY facts that will matter in future conversations
- Do NOT save: code patterns visible in the repo, file layouts, git history, temporary task state, debugging solutions already in code
- Convert relative dates to absolute dates (e.g., "next Thursday" → "2026-04-09")
- For feedback memories: capture the rule, WHY it exists, and HOW to apply it
- For project memories: capture the fact, WHY it matters, and HOW it affects decisions
- Prefer UPDATING an existing memory over creating a duplicate
- If nothing worth saving exists, return an empty list

## Existing Memories
{existing_memories}

## Recent Conversation
{recent_messages}

## Output Format
Return a JSON array of memory objects. Each object has:
- "action": "create" or "update" or "delete"
- "id": (only for update/delete) the existing memory ID
- "title": short name
- "type": one of "user", "feedback", "project", "reference"
- "scope": one of "user", "assistant", "project", "team"
- "summary": one-line description
- "body": full content

Respond with ONLY the JSON array, no other text. If nothing to save, respond with [].
"""


class AutoMemoryExtractor:
    """Identifies and persists durable facts from recent conversation."""

    def __init__(self, memory_manager: PersistentMemoryManager, llm: Any) -> None:
        super().__init__()
        self._memory = memory_manager
        self._llm = llm

    async def extract(
        self,
        recent_messages: list[Any],
        scope: MemoryScope = MemoryScope.USER,
        agent_wrote_memory_this_turn: bool = False,
    ) -> list[MemoryRecord]:
        """Extract and persist memories from recent messages.

        Returns the list of created/updated records.
        Skips extraction if the agent already wrote memories this turn.
        """
        if agent_wrote_memory_this_turn:
            logger.debug("Skipping extraction: agent wrote memory this turn")
            return []

        if not recent_messages:
            return []

        # Load existing memories for dedup
        existing = await self._memory.list_by_scope(scope, limit=50)
        existing_text = self._format_existing(existing)
        messages_text = self._format_messages(recent_messages)

        prompt = _EXTRACTION_PROMPT.format(
            existing_memories=existing_text or "(none)",
            recent_messages=messages_text,
        )

        # Call LLM for extraction (text-only, no tools)
        try:
            from langchain_core.messages import HumanMessage

            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception:
            logger.exception("Memory extraction LLM call failed")
            return []

        return await self._process_candidates(raw, scope)

    def _format_existing(self, records: list[MemoryRecord]) -> str:
        lines: list[str] = []
        for r in records:
            lines.append(f"- [{r.type.value}] {r.title}: {r.summary} (id: {r.id})")
        return "\n".join(lines)

    def _format_messages(self, messages: list[Any]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            if isinstance(content, str) and content.strip():
                # Truncate very long messages
                if len(content) > _MAX_MESSAGE_CHARS:
                    content = content[:_MAX_MESSAGE_CHARS] + "..."
                lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    async def _process_candidates(
        self, raw: str, scope: MemoryScope
    ) -> list[MemoryRecord]:
        """Parse LLM output and apply create/update/delete actions."""
        candidates = self._parse_response(raw)
        results: list[MemoryRecord] = []

        for candidate in candidates:
            try:
                action = candidate.get("action", "create")

                if action == "delete":
                    record_id = candidate.get("id", "")
                    if record_id:
                        await self._memory.delete(record_id, scope)
                    continue

                if action == "update":
                    record_id = candidate.get("id", "")
                    if record_id:
                        updated = await self._memory.update(
                            record_id,
                            scope,
                            {
                                "body": candidate.get("body", ""),
                                "summary": candidate.get("summary", ""),
                                "title": candidate.get("title", ""),
                            },
                        )
                        if updated:
                            results.append(updated)
                    continue

                # action == "create"
                record = MemoryRecord(
                    title=candidate.get("title", "Untitled"),
                    type=MemoryType(candidate.get("type", "user")),
                    scope=scope,
                    summary=candidate.get("summary", ""),
                    body=candidate.get("body", ""),
                    source="auto_extraction",
                )
                saved = await self._memory.create(record)
                results.append(saved)

            except Exception:
                logger.exception(
                    "Failed to process extraction candidate: %s", candidate
                )
                continue

        return results

    def _parse_response(self, raw: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response."""
        text = raw.strip()

        # Try direct JSON parse
        try:
            parsed: Any = json.loads(text)
            if isinstance(parsed, list):
                return parsed  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from surrounding text
        match = _RE_JSON_ARRAY.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse extraction response")
        return []
