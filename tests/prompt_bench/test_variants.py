"""Tests for variant loading, section overlay, and middleware patching."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.prompt_bench.variants import (
    PromptOverlay,
    discover_variants,
    load_variant,
    overlay_from_variant_file,
    patch_middleware_prompts,
)


class TestLoadVariant:
    def test_strips_yaml_frontmatter(self, tmp_path: Path) -> None:
        path = tmp_path / "v.md"
        path.write_text(
            "---\ntarget: x\nnotes: header\n---\nActual prompt body here.\n"
        )
        assert load_variant(path) == "Actual prompt body here."

    def test_passes_through_when_no_frontmatter(self, tmp_path: Path) -> None:
        path = tmp_path / "v.md"
        path.write_text("Plain body.\n")
        assert load_variant(path) == "Plain body."


class TestDiscoverVariants:
    def test_discovers_seed_variants(self) -> None:
        root = Path(__file__).parent
        variants = discover_variants(root, target="reference_deep_agent.core_identity")
        # baseline.md + deliberately_broken.md
        assert "baseline" in variants
        assert "deliberately_broken" in variants


class TestPatchMiddleware:
    def test_patches_and_restores(self) -> None:
        # Use the actual extraction module's _EXTRACTION_PROMPT to prove
        # the round-trip works against real kit code.
        from langgraph_kit.core.memory import extraction

        original = extraction._EXTRACTION_PROMPT
        overlay = PromptOverlay(
            name="t",
            middleware_overrides={
                "langgraph_kit.core.memory.extraction:_EXTRACTION_PROMPT": "REPLACED",
            },
        )
        with patch_middleware_prompts(overlay):
            assert extraction._EXTRACTION_PROMPT == "REPLACED"
        assert original == extraction._EXTRACTION_PROMPT

    def test_unknown_attr_raises(self) -> None:
        overlay = PromptOverlay(
            name="t",
            middleware_overrides={
                "langgraph_kit.core.memory.extraction:DEFINITELY_NOT_AN_ATTR": "x",
            },
        )
        with (
            pytest.raises(AttributeError, match="DEFINITELY_NOT_AN_ATTR"),
            patch_middleware_prompts(overlay),
        ):
            pass

    def test_dot_separator_works_too(self) -> None:
        from langgraph_kit.core.memory import extraction

        original = extraction._EXTRACTION_PROMPT
        overlay = PromptOverlay(
            name="t",
            middleware_overrides={
                "langgraph_kit.core.memory.extraction._EXTRACTION_PROMPT": "DOT_REPLACED",
            },
        )
        with patch_middleware_prompts(overlay):
            assert extraction._EXTRACTION_PROMPT == "DOT_REPLACED"
        assert original == extraction._EXTRACTION_PROMPT


class TestOverlayFromVariantFile:
    def test_section_overlay(self) -> None:
        overlay = overlay_from_variant_file(
            name="x", section_id="core_identity", text="hi"
        )
        assert overlay.section_overrides == {"core_identity": "hi"}
        assert overlay.middleware_overrides == {}

    def test_middleware_overlay(self) -> None:
        overlay = overlay_from_variant_file(
            name="x",
            middleware_attr="langgraph_kit.core.memory.extraction:_EXTRACTION_PROMPT",
            text="hi",
        )
        assert overlay.middleware_overrides == {
            "langgraph_kit.core.memory.extraction:_EXTRACTION_PROMPT": "hi"
        }

    def test_requires_exactly_one(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            overlay_from_variant_file(name="x", text="hi")
        with pytest.raises(ValueError, match="exactly one"):
            overlay_from_variant_file(
                name="x",
                section_id="a",
                middleware_attr="b:c",
                text="hi",
            )
