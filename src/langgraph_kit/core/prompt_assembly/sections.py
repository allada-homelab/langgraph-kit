"""Prompt section model and registry for layered prompt composition."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator


class SectionStability(StrEnum):
    STABLE = "stable"
    VOLATILE = "volatile"
    CONDITIONAL = "conditional"


class PromptSection(BaseModel):
    """A single prompt section with stability metadata.

    Immutable by design: the ``cache_key`` is a content hash computed at
    construction, so allowing post-construction mutation of ``content``
    would leave the cache key stale and silently serve old content from
    the section cache (Area 2c in the review). To edit a section, build
    a new one with the new content.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    stability: SectionStability
    priority: int = 0
    condition: str | None = None
    cache_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _auto_cache_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("cache_key") is None:
            content = data.get("content", "")
            if not isinstance(content, str):
                return data
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            data["cache_key"] = f"{data.get('id', '')}:{content_hash}"
        return data


class SectionRegistry:
    """Registry for prompt sections, supporting conditional filtering."""

    def __init__(self) -> None:
        super().__init__()
        self._sections: dict[str, PromptSection] = {}

    def register(self, section: PromptSection) -> None:
        self._sections[section.id] = section

    def register_many(self, sections: Sequence[PromptSection]) -> None:
        for section in sections:
            self.register(section)

    def get(self, section_id: str) -> PromptSection | None:
        return self._sections.get(section_id)

    def get_active(self, conditions: set[str] | None = None) -> list[PromptSection]:
        """Return sections filtered by stability and active conditions.

        STABLE and VOLATILE sections are always included. CONDITIONAL sections
        are included only when their condition key is present in the conditions set.
        Results are sorted by priority descending.
        """
        active_conditions = conditions or set()
        result: list[PromptSection] = []
        for section in self._sections.values():
            if section.stability in (
                SectionStability.STABLE,
                SectionStability.VOLATILE,
            ) or (
                section.stability == SectionStability.CONDITIONAL
                and section.condition is not None
                and section.condition in active_conditions
            ):
                result.append(section)
        return sorted(result, key=lambda s: s.priority, reverse=True)

    def get_by_stability(self, stability: SectionStability) -> list[PromptSection]:
        return [s for s in self._sections.values() if s.stability == stability]

    def remove(self, section_id: str) -> None:
        self._sections.pop(section_id, None)
