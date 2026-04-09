"""Prompt section cache model for separating stable from volatile content."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)


class CacheEntry(BaseModel):
    section_id: str
    cache_key: str
    content: str
    stability: SectionStability
    cached_at: datetime


class CacheChangeRecord(BaseModel):
    section_id: str
    change_type: Literal["added", "updated", "removed", "unchanged"]
    reason: str


class PromptCache:
    """Cache for prompt sections, reusing stable content across compositions."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[str, CacheEntry] = {}
        self._previous_section_ids: set[str] = set()
        self._changes: list[CacheChangeRecord] = []

    def get_or_compute(self, section: PromptSection) -> tuple[str, bool]:
        """Return (content, was_cached).

        STABLE sections are cached and reused until invalidated.
        VOLATILE sections always return fresh content (never cached).
        CONDITIONAL sections are cached while their condition is active.
        """
        if section.stability == SectionStability.VOLATILE:
            return section.content, False

        existing = self._entries.get(section.id)
        if existing is not None and existing.cache_key == section.cache_key:
            return existing.content, True

        # Cache miss or key changed — store new entry
        self._entries[section.id] = CacheEntry(
            section_id=section.id,
            cache_key=section.cache_key or section.id,
            content=section.content,
            stability=section.stability,
            cached_at=datetime.now(UTC),
        )
        return section.content, False

    def invalidate(self, section_id: str) -> None:
        self._entries.pop(section_id, None)

    def invalidate_all(self) -> None:
        self._entries.clear()

    def get_changes_since_last_compose(self) -> list[CacheChangeRecord]:
        return list(self._changes)

    def compose_with_cache(
        self, sections: list[PromptSection]
    ) -> tuple[str, list[CacheChangeRecord]]:
        """Compose sections using cache. Returns (prompt, change_records)."""
        changes: list[CacheChangeRecord] = []
        current_ids: set[str] = set()
        parts: list[str] = []

        for section in sections:
            current_ids.add(section.id)
            content, was_cached = self.get_or_compute(section)
            parts.append(content)

            if section.id not in self._previous_section_ids:
                changes.append(
                    CacheChangeRecord(
                        section_id=section.id,
                        change_type="added",
                        reason="new section",
                    )
                )
            elif was_cached:
                changes.append(
                    CacheChangeRecord(
                        section_id=section.id,
                        change_type="unchanged",
                        reason="cache hit",
                    )
                )
            else:
                changes.append(
                    CacheChangeRecord(
                        section_id=section.id,
                        change_type="updated",
                        reason="content or cache key changed",
                    )
                )

        for removed_id in self._previous_section_ids - current_ids:
            changes.append(
                CacheChangeRecord(
                    section_id=removed_id,
                    change_type="removed",
                    reason="section no longer active",
                )
            )
            self._entries.pop(removed_id, None)

        self._previous_section_ids = current_ids
        self._changes = changes
        return "\n\n".join(parts), changes
