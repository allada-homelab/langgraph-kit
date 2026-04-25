"""Tests for the profile bridge — section overlay application."""

from __future__ import annotations

import pytest

from langgraph_kit.core.prompt_assembly.sections import SectionStability
from tests.prompt_bench.profiles import (
    apply_overlay_to_sections,
    get_baseline_sections,
)
from tests.prompt_bench.variants import PromptOverlay


class TestGetBaselineSections:
    def test_returns_fresh_copy(self) -> None:
        a = get_baseline_sections("reference_deep_agent")
        b = get_baseline_sections("reference_deep_agent")
        assert a == b
        assert a is not b  # caller can mutate without affecting source

    def test_unknown_profile(self) -> None:
        with pytest.raises(KeyError, match="Unknown agent profile"):
            get_baseline_sections("nonexistent_profile")


class TestApplyOverlayToSections:
    def test_preserves_metadata_when_swapping(self) -> None:
        baseline = get_baseline_sections("reference_deep_agent")
        original = next(s for s in baseline if s.id == "core_identity")

        overlay = PromptOverlay(
            name="test",
            section_overrides={"core_identity": "REPLACEMENT TEXT"},
        )
        swapped = apply_overlay_to_sections("reference_deep_agent", overlay)
        new_section = next(s for s in swapped if s.id == "core_identity")

        assert new_section.content == "REPLACEMENT TEXT"
        assert new_section.stability == original.stability
        assert new_section.priority == original.priority
        assert new_section.condition == original.condition

    def test_returns_baseline_when_no_overrides(self) -> None:
        overlay = PromptOverlay(name="t")
        result = apply_overlay_to_sections("reference_deep_agent", overlay)
        baseline = get_baseline_sections("reference_deep_agent")
        assert [s.id for s in result] == [s.id for s in baseline]
        assert [s.content for s in result] == [s.content for s in baseline]

    def test_unknown_section_id_raises(self) -> None:
        overlay = PromptOverlay(
            name="t",
            section_overrides={"this_section_does_not_exist": "x"},
        )
        with pytest.raises(KeyError, match="this_section_does_not_exist"):
            apply_overlay_to_sections("reference_deep_agent", overlay)

    def test_only_targeted_section_changes(self) -> None:
        baseline = get_baseline_sections("reference_deep_agent")
        baseline_by_id = {s.id: s for s in baseline}

        overlay = PromptOverlay(
            name="t",
            section_overrides={"core_identity": "NEW"},
        )
        swapped = apply_overlay_to_sections("reference_deep_agent", overlay)
        for section in swapped:
            if section.id == "core_identity":
                assert section.content == "NEW"
            else:
                assert section.content == baseline_by_id[section.id].content

    def test_conditional_section_overlay_preserves_condition(self) -> None:
        # ``memory_instructions`` is CONDITIONAL with condition="memory" —
        # an overlay must not flatten that to STABLE.
        overlay = PromptOverlay(
            name="t",
            section_overrides={"memory_instructions": "swapped memory text"},
        )
        swapped = apply_overlay_to_sections("reference_deep_agent", overlay)
        new_section = next(s for s in swapped if s.id == "memory_instructions")
        assert new_section.stability == SectionStability.CONDITIONAL
        assert new_section.condition == "memory"
