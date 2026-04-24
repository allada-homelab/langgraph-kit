"""Post-turn automatic memory extraction from recent conversation."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.internal_tags import (
    MEMORY_EXTRACTION_TAG,
    internal_llm_config,
)
from langgraph_kit.core.memory._parsing import parse_json_array
from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    coerce_memory_type,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

logger = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 2000  # truncate messages longer than this when formatting for LLM

# Upper bound on how many candidates a single extraction pass may apply.
# Guards against a runaway or malicious LLM writing dozens of records in
# one turn. Can be overridden via ``AutoMemoryExtractor(..., max_candidates=N)``.
_DEFAULT_MAX_CANDIDATES = 10


def _coerce_memory_scope(value: Any) -> MemoryScope | None:
    """Best-effort scope coercion that rejects unknown strings."""
    if isinstance(value, MemoryScope):
        return value
    if not isinstance(value, str):
        return None
    try:
        return MemoryScope(value)
    except ValueError:
        return None

_EXTRACTION_PROMPT = """You are a memory extraction worker. Your ONLY job is to identify durable, future-useful facts from the recent conversation that should be saved to long-term memory.

Today's date: {today}

## Rules
- Save ONLY facts that will matter in future conversations
- Do NOT save: code patterns visible in the repo, file layouts, git history, temporary task state, debugging solutions already in code
- Convert relative dates to absolute dates using today's date above
- For feedback memories: capture the rule, WHY it exists, and HOW to apply it
- For project memories: capture the fact, WHY it matters, and HOW it affects decisions
- Prefer UPDATING an existing memory over creating a duplicate
- If nothing worth saving exists, return an empty list

## Memory Types
- "user": personal preferences, role, communication style, constraints
- "feedback": correction or rule from the user about how to work ("always X", "never Y")
- "project": project-specific facts (tech stack, deploy targets, naming conventions)
- "reference": external links, API key locations, doc URLs, tool names

## Scopes
- "user": visible only to this user
- "project": visible to anyone working on this project
- "team": visible to the whole team
- "assistant": internal to this assistant instance

## Existing Memories
{existing_memories}

## Recent Conversation
{recent_messages}

## Output Format
Return a JSON array. Each object has:
- "action": "create" | "update" | "delete"
- "id": (only for update/delete) the existing memory ID
- "title": short descriptive name
- "type": one of "user", "feedback", "project", "reference"
- "scope": one of "user", "assistant", "project", "team"
- "summary": one-line description
- "body": full content

## Example
[
  {{"action": "create", "title": "Prefers ruff over black", "type": "feedback", "scope": "user", "summary": "User corrected formatter choice", "body": "Always use ruff for formatting. User dislikes black's opinionated wrapping."}},
  {{"action": "update", "id": "mem_abc", "title": "Deploy target", "type": "project", "scope": "project", "summary": "Deploy target changed", "body": "Production deploy moved from ECS to K8s as of 2026-03."}}
]

Respond with ONLY the JSON array. If nothing to save, respond with [].
"""


class AutoMemoryExtractor:
    """Identifies and persists durable facts from recent conversation."""

    def __init__(
        self,
        memory_manager: PersistentMemoryManager,
        llm: Any,
        *,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        super().__init__()
        self._memory = memory_manager
        self._llm = llm
        self._max_candidates = max_candidates

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

        from datetime import date

        prompt = _EXTRACTION_PROMPT.format(
            today=date.today().isoformat(),
            existing_memories=existing_text or "(none)",
            recent_messages=messages_text,
        )

        # Call LLM for extraction (text-only, no tools). The call is tagged so
        # that consumers streaming via astream_events can filter the emitted
        # chat_model events out of the user-facing transcript — see
        # langgraph_kit.core.internal_tags for rationale.
        try:
            from langchain_core.messages import HumanMessage

            response = await self._llm.ainvoke(
                [HumanMessage(content=prompt)],
                config=internal_llm_config(
                    MEMORY_EXTRACTION_TAG, run_name="memory_extraction"
                ),
            )
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
        # Cap how many records a single extraction call can produce —
        # a runaway LLM could otherwise write dozens per turn.
        if len(candidates) > self._max_candidates:
            logger.warning(
                "Extraction returned %d candidates; capping at %d.",
                len(candidates),
                self._max_candidates,
            )
            candidates = candidates[: self._max_candidates]

        results: list[MemoryRecord] = []

        for candidate in candidates:
            try:
                action = candidate.get("action", "create")

                # Honour the per-candidate scope when the LLM names one we
                # recognise; otherwise fall back to the caller's scope. The
                # prompt instructs the model to pick a scope, so ignoring
                # it wholesale (as the prior code did) was a contract gap.
                effective_scope = (
                    _coerce_memory_scope(candidate.get("scope")) or scope
                )

                if action == "delete":
                    record_id = candidate.get("id", "")
                    if record_id:
                        await self._memory.delete(record_id, effective_scope)
                    continue

                if action == "update":
                    record_id = candidate.get("id", "")
                    if record_id:
                        updates: dict[str, Any] = {
                            "body": candidate.get("body", ""),
                            "summary": candidate.get("summary", ""),
                            "title": candidate.get("title", ""),
                        }
                        # Allow the LLM to re-classify a mis-typed record.
                        mem_type = coerce_memory_type(candidate.get("type"))
                        if mem_type is not None:
                            updates["type"] = mem_type
                        updated = await self._memory.update(
                            record_id,
                            effective_scope,
                            updates,
                        )
                        if updated:
                            results.append(updated)
                    continue

                # action == "create"
                mem_type = coerce_memory_type(candidate.get("type"))
                if mem_type is None:
                    # Extractor LLMs sometimes invent enum members (e.g.
                    # "assistant", "system", "note"). Skip rather than crash
                    # the batch or silently coerce to a wrong default.
                    logger.warning(
                        "Extraction candidate has unrecognized type=%r; skipping. Candidate title=%r",
                        candidate.get("type"),
                        candidate.get("title"),
                    )
                    continue
                record = MemoryRecord(
                    title=candidate.get("title", "Untitled"),
                    type=mem_type,
                    scope=effective_scope,
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
        return parse_json_array(raw, context="extraction response")
