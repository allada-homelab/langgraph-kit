"""Tests for prompt assembly module."""

from __future__ import annotations

import pytest

from langgraph_kit.core.prompt_assembly.cache import PromptCache
from langgraph_kit.core.prompt_assembly.composer import PromptComposer
from langgraph_kit.core.prompt_assembly.context_providers import (
    MemoryContextProvider,
    ThreadContextProvider,
    ToolContextProvider,
)
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionRegistry,
    SectionStability,
)

# ---------------------------------------------------------------------------
# PromptSection
# ---------------------------------------------------------------------------


class TestPromptSection:
    def test_section_auto_cache_key(self) -> None:
        section = PromptSection(
            id="intro",
            content="Hello world",
            stability=SectionStability.STABLE,
        )
        assert section.cache_key is not None
        assert section.cache_key.startswith("intro:")
        assert len(section.cache_key) > len("intro:")

    def test_section_explicit_cache_key(self) -> None:
        section = PromptSection(
            id="intro",
            content="Hello world",
            stability=SectionStability.STABLE,
            cache_key="my-custom-key",
        )
        assert section.cache_key == "my-custom-key"


# ---------------------------------------------------------------------------
# SectionRegistry
# ---------------------------------------------------------------------------


class TestSectionRegistry:
    def _make_section(
        self,
        section_id: str = "s1",
        content: str = "content",
        stability: SectionStability = SectionStability.STABLE,
        priority: int = 0,
        condition: str | None = None,
    ) -> PromptSection:
        return PromptSection(
            id=section_id,
            content=content,
            stability=stability,
            priority=priority,
            condition=condition,
        )

    def test_register_and_get(self) -> None:
        registry = SectionRegistry()
        section = self._make_section()
        registry.register(section)
        assert registry.get("s1") is section
        assert registry.get("nonexistent") is None

    def test_register_many(self) -> None:
        registry = SectionRegistry()
        sections = [
            self._make_section(section_id="a"),
            self._make_section(section_id="b"),
        ]
        registry.register_many(sections)
        assert registry.get("a") is not None
        assert registry.get("b") is not None

    def test_get_active_stable_always_included(self) -> None:
        registry = SectionRegistry()
        registry.register(self._make_section(stability=SectionStability.STABLE))
        active = registry.get_active()
        assert len(active) == 1
        assert active[0].id == "s1"

    def test_get_active_volatile_always_included(self) -> None:
        registry = SectionRegistry()
        registry.register(self._make_section(stability=SectionStability.VOLATILE))
        active = registry.get_active()
        assert len(active) == 1
        assert active[0].stability == SectionStability.VOLATILE

    def test_get_active_conditional_with_matching_condition(self) -> None:
        registry = SectionRegistry()
        registry.register(
            self._make_section(
                stability=SectionStability.CONDITIONAL, condition="has_tools"
            )
        )
        active = registry.get_active(conditions={"has_tools"})
        assert len(active) == 1

    def test_get_active_conditional_without_condition(self) -> None:
        registry = SectionRegistry()
        registry.register(
            self._make_section(
                stability=SectionStability.CONDITIONAL, condition="has_tools"
            )
        )
        active = registry.get_active()
        assert len(active) == 0

    def test_get_active_sorted_by_priority(self) -> None:
        registry = SectionRegistry()
        registry.register_many(
            [
                self._make_section(section_id="low", priority=1),
                self._make_section(section_id="high", priority=10),
                self._make_section(section_id="mid", priority=5),
            ]
        )
        active = registry.get_active()
        assert [s.id for s in active] == ["high", "mid", "low"]

    def test_remove_section(self) -> None:
        registry = SectionRegistry()
        registry.register(self._make_section())
        registry.remove("s1")
        assert registry.get("s1") is None


# ---------------------------------------------------------------------------
# PromptComposer
# ---------------------------------------------------------------------------


class TestPromptComposer:
    def test_compose_sections_only(self) -> None:
        registry = SectionRegistry()
        registry.register_many(
            [
                PromptSection(
                    id="a", content="Part A", stability=SectionStability.STABLE
                ),
                PromptSection(
                    id="b", content="Part B", stability=SectionStability.STABLE
                ),
            ]
        )
        composer = PromptComposer(registry)
        result = composer.compose_sections_only()
        assert "Part A" in result
        assert "Part B" in result

    @pytest.mark.asyncio
    async def test_compose_with_conditions(self) -> None:
        registry = SectionRegistry()
        registry.register_many(
            [
                PromptSection(
                    id="always",
                    content="Always here",
                    stability=SectionStability.STABLE,
                ),
                PromptSection(
                    id="cond",
                    content="Conditional content",
                    stability=SectionStability.CONDITIONAL,
                    condition="feature_x",
                ),
            ]
        )
        composer = PromptComposer(registry)

        without = await composer.compose(conditions=None)
        assert "Conditional content" not in without

        with_cond = await composer.compose(conditions={"feature_x"})
        assert "Conditional content" in with_cond
        assert "Always here" in with_cond

    @pytest.mark.asyncio
    async def test_compose_with_providers(self) -> None:
        registry = SectionRegistry()
        registry.register(
            PromptSection(
                id="base", content="Base prompt", stability=SectionStability.STABLE
            )
        )
        provider = ThreadContextProvider()
        composer = PromptComposer(registry, providers=[provider])
        result = await composer.compose(
            context={"thread_id": "t-123", "message_count": 5}
        )
        assert "Base prompt" in result
        assert "Thread: t-123" in result
        assert "Messages: 5" in result

    def test_get_active_section_ids(self) -> None:
        registry = SectionRegistry()
        registry.register_many(
            [
                PromptSection(
                    id="x", content="X", stability=SectionStability.STABLE, priority=1
                ),
                PromptSection(
                    id="y", content="Y", stability=SectionStability.STABLE, priority=2
                ),
            ]
        )
        composer = PromptComposer(registry)
        ids = composer.get_active_section_ids()
        assert ids == ["y", "x"]


# ---------------------------------------------------------------------------
# ContextProviders
# ---------------------------------------------------------------------------


class TestContextProviders:
    @pytest.mark.asyncio
    async def test_thread_context_provider(self) -> None:
        provider = ThreadContextProvider()
        result = await provider.provide({"thread_id": "abc", "message_count": 3})
        assert "Thread: abc" in result
        assert "Messages: 3" in result

    @pytest.mark.asyncio
    async def test_memory_context_provider_with_memories(self) -> None:
        provider = MemoryContextProvider()
        result = await provider.provide(
            {"memories": ["User likes Python", "Prefers dark mode"]}
        )
        assert "# Relevant Memory" in result
        assert "- User likes Python" in result
        assert "- Prefers dark mode" in result

    @pytest.mark.asyncio
    async def test_memory_context_provider_empty(self) -> None:
        provider = MemoryContextProvider()
        result = await provider.provide({})
        assert result == ""

    @pytest.mark.asyncio
    async def test_tool_context_provider(self) -> None:
        provider = ToolContextProvider()
        result = await provider.provide(
            {"tool_guidance": ["Use search first", "Prefer read over grep"]}
        )
        assert "# Tool Guidance" in result
        assert "- Use search first" in result
        assert "- Prefer read over grep" in result


# ---------------------------------------------------------------------------
# PromptCache
# ---------------------------------------------------------------------------


class TestPromptCache:
    def test_stable_section_cached(self) -> None:
        cache = PromptCache()
        section = PromptSection(
            id="s", content="hello", stability=SectionStability.STABLE
        )
        content1, was_cached1 = cache.get_or_compute(section)
        assert content1 == "hello"
        assert was_cached1 is False

        content2, was_cached2 = cache.get_or_compute(section)
        assert content2 == "hello"
        assert was_cached2 is True

    def test_volatile_section_not_cached(self) -> None:
        cache = PromptCache()
        section = PromptSection(
            id="v", content="dynamic", stability=SectionStability.VOLATILE
        )
        _, was_cached1 = cache.get_or_compute(section)
        assert was_cached1 is False

        _, was_cached2 = cache.get_or_compute(section)
        assert was_cached2 is False

    def test_invalidate_section(self) -> None:
        cache = PromptCache()
        section = PromptSection(
            id="s", content="hello", stability=SectionStability.STABLE
        )
        cache.get_or_compute(section)
        cache.invalidate("s")

        _, was_cached = cache.get_or_compute(section)
        assert was_cached is False

    def test_compose_with_cache_tracks_changes(self) -> None:
        cache = PromptCache()
        sections = [
            PromptSection(id="a", content="A", stability=SectionStability.STABLE),
            PromptSection(id="b", content="B", stability=SectionStability.VOLATILE),
        ]

        # First compose — both are new
        prompt1, changes1 = cache.compose_with_cache(sections)
        assert "A" in prompt1
        assert "B" in prompt1
        assert len(changes1) == 2
        assert all(c.change_type == "added" for c in changes1)

        # Second compose — stable is cached (unchanged), volatile is updated
        prompt2, changes2 = cache.compose_with_cache(sections)
        assert prompt2 == prompt1
        change_map = {c.section_id: c.change_type for c in changes2}
        assert change_map["a"] == "unchanged"
        # Volatile always re-computed, but id is not new so it's "updated"
        assert change_map["b"] == "updated"


# ---------------------------------------------------------------------------
# Versioning — issue #18.
# ---------------------------------------------------------------------------


class TestPromptVersioning:
    @staticmethod
    def _section(
        section_id: str = "core_role",
        content: str = "v1 content",
        version: str = "1",
        priority: int = 0,
        stability: SectionStability = SectionStability.STABLE,
        condition: str | None = None,
    ) -> PromptSection:
        return PromptSection(
            id=section_id,
            content=content,
            version=version,
            priority=priority,
            stability=stability,
            condition=condition,
        )

    @staticmethod
    def _require(
        registry: SectionRegistry,
        section_id: str,
        version: str | None = None,
    ) -> PromptSection:
        """Get-or-fail wrapper so test assertions can read attributes
        on the result without basedpyright Optional-narrowing noise.
        """
        section = registry.get(section_id, version=version)
        assert section is not None, (
            f"expected section {section_id!r} version {version!r} to be registered"
        )
        return section

    def test_default_version_is_one_string(self) -> None:
        section = self._section()
        assert section.version == "1"

    def test_version_round_trips_through_serialization(self) -> None:
        original = self._section(version="2026-04-15")
        round_tripped = PromptSection.model_validate_json(original.model_dump_json())
        assert round_tripped.version == "2026-04-15"

    def test_register_promotes_first_version_implicitly(self) -> None:
        registry = SectionRegistry()
        v1 = self._section(version="1")
        registry.register(v1)
        assert registry.current_version("core_role") == "1"
        assert self._require(registry, "core_role") is v1

    def test_register_with_set_current_false_keeps_existing_current(self) -> None:
        """Staging a candidate version doesn't auto-promote it."""
        registry = SectionRegistry()
        v1 = self._section(version="1", content="live")
        v2 = self._section(version="2", content="staged")
        registry.register(v1)
        registry.register(v2, set_current=False)
        assert registry.current_version("core_role") == "1"
        assert self._require(registry, "core_role").content == "live"
        assert self._require(registry, "core_role", version="2").content == "staged"
        assert registry.list_versions("core_role") == ["1", "2"]

    def test_register_default_promotes_new_version(self) -> None:
        """Plain register() of a new version makes it current."""
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="old"))
        registry.register(self._section(version="2", content="new"))
        assert registry.current_version("core_role") == "2"
        assert self._require(registry, "core_role").content == "new"

    def test_set_current_switches_active_version(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="v1"))
        registry.register(self._section(version="2", content="v2"), set_current=False)
        assert self._require(registry, "core_role").content == "v1"
        registry.set_current("core_role", "2")
        assert self._require(registry, "core_role").content == "v2"

    def test_set_current_unknown_id_raises(self) -> None:
        registry = SectionRegistry()
        with pytest.raises(KeyError, match="Unknown section id"):
            registry.set_current("missing", "1")

    def test_set_current_unknown_version_raises(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(version="1"))
        with pytest.raises(KeyError, match="Unknown version 'nope'"):
            registry.set_current("core_role", "nope")

    def test_get_explicit_version_bypasses_current(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="v1"))
        registry.register(self._section(version="2", content="v2"))
        assert self._require(registry, "core_role", version="1").content == "v1"
        assert self._require(registry, "core_role", version="2").content == "v2"

    def test_get_unknown_id_returns_none(self) -> None:
        assert SectionRegistry().get("missing") is None

    def test_get_unknown_version_returns_none(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(version="1"))
        assert registry.get("core_role", version="99") is None

    def test_get_active_uses_current_version_only(self) -> None:
        """get_active() must not return every version — only the current one."""
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="v1"))
        registry.register(self._section(version="2", content="v2"))
        active = registry.get_active()
        assert len(active) == 1
        assert active[0].content == "v2"
        registry.set_current("core_role", "1")
        active = registry.get_active()
        assert len(active) == 1
        assert active[0].content == "v1"

    def test_get_active_respects_conditions_per_current_version(self) -> None:
        """Conditional gating still works after version switching."""
        registry = SectionRegistry()
        registry.register(
            self._section(
                version="1",
                stability=SectionStability.CONDITIONAL,
                condition="memory",
            )
        )
        assert registry.get_active(conditions=set()) == []
        active = registry.get_active(conditions={"memory"})
        assert len(active) == 1

    def test_current_versions_snapshot(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(section_id="a", version="1"))
        registry.register(self._section(section_id="b", version="2"))
        snapshot = registry.current_versions()
        assert snapshot == {"a": "1", "b": "2"}
        # Snapshot is a copy — mutating it doesn't affect the registry.
        snapshot["a"] = "999"
        assert registry.current_version("a") == "1"

    def test_remove_specific_version_keeps_others(self) -> None:
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="v1"))
        registry.register(self._section(version="2", content="v2"))
        registry.remove("core_role", version="1")
        assert registry.list_versions("core_role") == ["2"]
        assert self._require(registry, "core_role").content == "v2"

    def test_remove_current_version_promotes_next(self) -> None:
        """Dropping the active version shouldn't leave the id orphaned."""
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="v1"))
        registry.register(self._section(version="2", content="v2"), set_current=False)
        # Currently pointing at "1"; remove it.
        registry.remove("core_role", version="1")
        assert registry.current_version("core_role") == "2"
        assert self._require(registry, "core_role").content == "v2"

    def test_remove_no_version_drops_entire_id(self) -> None:
        """Legacy ``remove(id)`` semantics: drop everything for that id."""
        registry = SectionRegistry()
        registry.register(self._section(version="1"))
        registry.register(self._section(version="2"))
        registry.remove("core_role")
        assert registry.list_versions("core_role") == []
        assert registry.current_version("core_role") is None

    def test_re_register_same_id_version_overwrites(self) -> None:
        """Same (id, version) re-registered replaces the prior content."""
        registry = SectionRegistry()
        registry.register(self._section(version="1", content="first"))
        registry.register(self._section(version="1", content="second"))
        assert self._require(registry, "core_role").content == "second"
        assert registry.list_versions("core_role") == ["1"]

    def test_legacy_single_version_callers_unchanged(self) -> None:
        """Callers that never set ``version`` see the same behavior as before."""
        registry = SectionRegistry()
        # Construct via the same kwargs the existing tests use — no version.
        section = PromptSection(
            id="core_role",
            content="hello",
            stability=SectionStability.STABLE,
        )
        registry.register(section)
        # get(id) still returns the section, list_versions has the default.
        assert self._require(registry, "core_role") is section
        assert registry.list_versions("core_role") == ["1"]
        assert registry.current_version("core_role") == "1"
