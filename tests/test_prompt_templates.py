"""Tests for the public ``langgraph_kit.prompt_templates`` library (issue #43 v1)."""

from __future__ import annotations

import pytest

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionRegistry,
    SectionStability,
)
from langgraph_kit.prompt_templates import (
    be_concise,
    completion_signal,
    core_identity,
    diff_section,
    error_handling,
    memory_awareness,
    operate_carefully,
    output_format_natural,
    output_format_structured,
    tool_use_discipline,
)

ALL_SHIPPED = [
    core_identity,
    operate_carefully,
    be_concise,
    completion_signal,
    memory_awareness,
    tool_use_discipline,
    error_handling,
    output_format_natural,
    output_format_structured,
]


# ---------------------------------------------------------------------------
# Library-shape invariants
# ---------------------------------------------------------------------------


class TestLibraryShape:
    def test_all_shipped_sections_are_promptsection_instances(self) -> None:
        for section in ALL_SHIPPED:
            assert isinstance(section, PromptSection), section

    def test_every_shipped_section_has_non_empty_content(self) -> None:
        for section in ALL_SHIPPED:
            assert section.content.strip(), f"empty content: {section.id}"

    def test_every_shipped_section_has_a_version(self) -> None:
        for section in ALL_SHIPPED:
            assert section.version, f"missing version: {section.id}"

    def test_section_ids_unique_within_a_version(self) -> None:
        """``output_format_natural`` and ``output_format_structured`` share an id but differ in version."""
        seen: set[tuple[str, str]] = set()
        for section in ALL_SHIPPED:
            key = (section.id, section.version)
            assert key not in seen, f"duplicate (id, version): {key}"
            seen.add(key)

    def test_at_least_eight_shipped_sections(self) -> None:
        """Issue #43 acceptance criterion: 8+ sections shipped."""
        assert len(ALL_SHIPPED) >= 8, len(ALL_SHIPPED)

    def test_conditional_sections_declare_their_condition(self) -> None:
        for section in ALL_SHIPPED:
            if section.stability == SectionStability.CONDITIONAL:
                assert section.condition, (
                    f"CONDITIONAL section {section.id} missing condition"
                )


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_register_many_with_full_library(self) -> None:
        """All shipped sections register cleanly into a fresh SectionRegistry."""
        registry = SectionRegistry()
        # Skip the duplicate-id pair to avoid auto-promotion confusion.
        registry.register_many(
            [s for s in ALL_SHIPPED if s.id != "output_format"]
            + [output_format_natural]  # explicitly pick one variant
        )
        # Spot-check: the registry sees every id we put in.
        for section in ALL_SHIPPED:
            if section.id == "output_format":
                continue
            assert registry.get(section.id) is not None, section.id

    def test_get_active_returns_unconditional_sections(self) -> None:
        """Without conditions set, only STABLE+VOLATILE sections appear."""
        registry = SectionRegistry()
        registry.register(core_identity)
        registry.register(memory_awareness)  # CONDITIONAL on "memory"
        active = registry.get_active(conditions=set())
        ids = [s.id for s in active]
        assert "core_identity" in ids
        assert "memory_awareness" not in ids

    def test_get_active_with_memory_condition_includes_memory_awareness(self) -> None:
        registry = SectionRegistry()
        registry.register(core_identity)
        registry.register(memory_awareness)
        active = registry.get_active(conditions={"memory"})
        ids = [s.id for s in active]
        assert "memory_awareness" in ids

    def test_overriding_a_shipped_section_replaces_it(self) -> None:
        """Per the docstring contract: register-by-id overrides whole-section."""
        registry = SectionRegistry()
        registry.register(core_identity)
        custom = core_identity.model_copy(
            update={"content": "Domain-specific identity here."}
        )
        registry.register(custom)
        result = registry.get("core_identity")
        assert result is not None
        assert result.content == "Domain-specific identity here."


# ---------------------------------------------------------------------------
# Versioning interplay (issue #18)
# ---------------------------------------------------------------------------


class TestVersioningInterplay:
    def test_output_format_pair_can_coexist_in_one_registry(self) -> None:
        """Same id, different versions — exactly what #18's per-id versioning enables."""
        registry = SectionRegistry()
        registry.register(output_format_natural, set_current=False)
        registry.register(output_format_structured)  # current
        assert sorted(registry.list_versions("output_format")) == sorted(
            ["1-natural", "1-structured"]
        )
        assert registry.current_version("output_format") == "1-structured"
        # Switch via #18's set_current API.
        registry.set_current("output_format", "1-natural")
        active = registry.get("output_format")
        assert active is not None
        assert "natural language" in active.content


# ---------------------------------------------------------------------------
# diff_section helper
# ---------------------------------------------------------------------------


class TestDiffSection:
    def test_identical_sections_diff_to_empty_string(self) -> None:
        assert diff_section(core_identity, core_identity) == ""

    def test_content_change_appears_in_diff(self) -> None:
        custom = core_identity.model_copy(update={"content": "Custom identity here."})
        diff = diff_section(custom, core_identity)
        assert "shipped:core_identity@1" in diff
        assert "Custom identity here." in diff
        assert diff.startswith("---")  # unified diff header

    def test_version_change_appears_in_diff(self) -> None:
        custom = core_identity.model_copy(update={"version": "custom-1"})
        diff = diff_section(custom, core_identity)
        assert "version: custom-1" in diff
        assert "version: 1" in diff

    def test_priority_change_appears_in_diff(self) -> None:
        custom = core_identity.model_copy(update={"priority": 999})
        diff = diff_section(custom, core_identity)
        assert "priority: 999" in diff

    def test_condition_change_appears_in_diff(self) -> None:
        custom = core_identity.model_copy(update={"condition": "premium-tier"})
        diff = diff_section(custom, core_identity)
        assert "condition: premium-tier" in diff

    def test_stability_change_appears_in_diff(self) -> None:
        custom = core_identity.model_copy(
            update={"stability": SectionStability.VOLATILE}
        )
        diff = diff_section(custom, core_identity)
        # Diff shows the actual enum value text.
        assert "stability: volatile" in diff

    def test_diff_uses_unified_format_markers(self) -> None:
        """Confirms the diff is parseable by terminal diff highlighters."""
        custom = core_identity.model_copy(update={"content": "x"})
        diff = diff_section(custom, core_identity)
        assert any(line.startswith("@@") for line in diff.splitlines())


# ---------------------------------------------------------------------------
# Public-API surface check
# ---------------------------------------------------------------------------


class TestPublicApi:
    def test_all_promised_names_importable(self) -> None:
        from langgraph_kit import prompt_templates

        for name in prompt_templates.__all__:
            assert hasattr(prompt_templates, name), name

    def test_diff_section_round_trip_no_change(self) -> None:
        """``diff_section(s, s)`` short-circuits without invoking difflib."""
        # Just confirms the equality fast-path produces the empty-string contract.
        for section in ALL_SHIPPED:
            assert diff_section(section, section) == ""


# ---------------------------------------------------------------------------
# pytest fixture for verifying section content stability
# ---------------------------------------------------------------------------


@pytest.fixture
def shipped_library() -> list[PromptSection]:
    """Convenience fixture for downstream test suites that want to scan the library."""
    return ALL_SHIPPED
