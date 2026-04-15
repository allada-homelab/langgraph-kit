"""Prompt section model and registry for layered prompt composition."""

from __future__ import annotations

import hashlib
from enum import Enum, StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from pydantic import BaseModel, model_validator


class SectionStability(StrEnum):
    STABLE = "stable"
    VOLATILE = "volatile"
    CONDITIONAL = "conditional"


class PromptSection(BaseModel):
    """A single prompt section with stability metadata."""

    id: str
    content: str
    stability: SectionStability
    priority: int = 0
    condition: str | None = None
    cache_key: str | None = None

    @model_validator(mode="after")
    def _auto_cache_key(self) -> PromptSection:
        if self.cache_key is None:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.cache_key = f"{self.id}:{content_hash}"
        return self


class SectionRegistry:
    """Registry for prompt sections, supporting conditional filtering."""

    def __init__(self) -> None:
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
