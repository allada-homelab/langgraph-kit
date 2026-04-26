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


DEFAULT_VERSION = "1"
"""Default ``PromptSection.version`` for sections that don't declare one.

Free-form string, not a SemVer or monotonic counter — version comparison
is intentionally up to the caller. ``"1"`` rather than ``"v1"`` keeps it
short for the common single-version case where the field is rarely
inspected.
"""


class PromptSection(BaseModel):
    """A single prompt section with stability metadata.

    Immutable by design: the ``cache_key`` is a content hash computed at
    construction, so allowing post-construction mutation of ``content``
    would leave the cache key stale and silently serve old content from
    the section cache (Area 2c in the review). To edit a section, build
    a new one with the new content.

    ``version`` is a free-form caller-controlled identifier (e.g.
    ``"v2"``, ``"2026-04-15"``, a git SHA). It's distinct from
    ``cache_key`` (which is a content hash and changes on any whitespace
    edit) — version is for *human* identification of "which prompt
    revision is live," not for cache invalidation. Multiple versions of
    a single section id can coexist in a :class:`SectionRegistry` and be
    swapped via :py:meth:`SectionRegistry.set_current`.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    stability: SectionStability
    priority: int = 0
    condition: str | None = None
    cache_key: str | None = None
    version: str = DEFAULT_VERSION

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
    """Registry for prompt sections, supporting conditional filtering and versioning.

    A section id may have multiple registered versions; one of them is
    "current" (the default for :py:meth:`get`, :py:meth:`get_active`, and
    legacy single-version reads). Switch the current version with
    :py:meth:`set_current`; introspect history with
    :py:meth:`list_versions`. The default :py:meth:`register` call sets
    the registered section as current — pass ``set_current=False`` to
    register a new version without promoting it (useful for staging a
    candidate while keeping the live version stable).

    The single-version code path is unchanged: callers that never set
    ``PromptSection.version`` register against the default
    :data:`DEFAULT_VERSION`, and ``get(id)`` returns the same object as
    before.
    """

    def __init__(self) -> None:
        super().__init__()
        # _versions[section_id][version] -> PromptSection
        self._versions: dict[str, dict[str, PromptSection]] = {}
        self._current: dict[str, str] = {}

    def register(self, section: PromptSection, *, set_current: bool = True) -> None:
        """Register *section*. Optionally promote it as the current version.

        Re-registering an existing ``(id, version)`` pair overwrites it
        — versions are caller-named and may be edited freely. Re-registering
        a different content hash under the same ``(id, version)`` is the
        caller's call (use ``"1.1"`` instead if you want history).
        """
        self._versions.setdefault(section.id, {})[section.version] = section
        if set_current or section.id not in self._current:
            self._current[section.id] = section.version

    def register_many(self, sections: Sequence[PromptSection]) -> None:
        for section in sections:
            self.register(section)

    def get(self, section_id: str, version: str | None = None) -> PromptSection | None:
        """Return the section for *section_id* at *version*.

        ``version=None`` returns whatever ``set_current`` last pointed at
        (defaults to the version registered first). Unknown ids and
        unknown versions both return ``None`` — callers that need
        existence-check-then-fetch should use :py:meth:`list_versions`
        first to disambiguate.
        """
        versions = self._versions.get(section_id)
        if not versions:
            return None
        if version is None:
            version = self._current.get(section_id)
            if version is None:
                return None
        return versions.get(version)

    def list_versions(self, section_id: str) -> list[str]:
        """Return the registered versions for *section_id* (insertion order)."""
        versions = self._versions.get(section_id)
        if not versions:
            return []
        return list(versions.keys())

    def current_version(self, section_id: str) -> str | None:
        """Return the version :py:meth:`get` would currently return."""
        return self._current.get(section_id)

    def current_versions(self) -> dict[str, str]:
        """Snapshot of every section_id → current-version mapping.

        Useful for tagging a run with its active prompt versions
        (e.g. as Langfuse metadata) without iterating the registry.
        """
        return dict(self._current)

    def set_current(self, section_id: str, version: str) -> None:
        """Promote *version* of *section_id* to current.

        Raises ``KeyError`` if either the id or version is unknown —
        silently swapping to a non-existent version would let typos
        de-activate a section without warning.
        """
        versions = self._versions.get(section_id)
        if not versions:
            msg = f"Unknown section id: {section_id!r}"
            raise KeyError(msg)
        if version not in versions:
            msg = (
                f"Unknown version {version!r} for section {section_id!r}; "
                f"known versions: {sorted(versions)}"
            )
            raise KeyError(msg)
        self._current[section_id] = version

    def get_active(self, conditions: set[str] | None = None) -> list[PromptSection]:
        """Return sections filtered by stability and active conditions.

        Each section id contributes its *current* version only — version
        switching is the only knob; this method is unaware of history.
        STABLE and VOLATILE sections are always included. CONDITIONAL
        sections are included only when their condition key is present
        in the conditions set. Results are sorted by priority descending.
        """
        active_conditions = conditions or set()
        result: list[PromptSection] = []
        for section_id in self._versions:
            section = self.get(section_id)
            if section is None:
                continue
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
        return [
            section
            for section_id in self._versions
            if (section := self.get(section_id)) is not None
            and section.stability == stability
        ]

    def remove(self, section_id: str, version: str | None = None) -> None:
        """Drop *section_id* (or one specific version of it).

        ``version=None`` removes every version of the id (the legacy
        single-version behavior). Removing the currently-pointed version
        promotes the next version in registration order, or clears the
        current-pointer entirely if none remain.
        """
        versions = self._versions.get(section_id)
        if not versions:
            return
        if version is None:
            self._versions.pop(section_id, None)
            self._current.pop(section_id, None)
            return
        versions.pop(version, None)
        if not versions:
            self._versions.pop(section_id, None)
            self._current.pop(section_id, None)
            return
        if self._current.get(section_id) == version:
            self._current[section_id] = next(iter(versions))
