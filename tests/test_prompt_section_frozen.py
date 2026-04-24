"""Regression: ``PromptSection`` is frozen so cache_key cannot go stale.

Before this fix, a constructed section could have its ``.content``
re-assigned, but ``cache_key`` was fixed at construction — which meant
the section cache served the old content despite the field having
changed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)


def test_cannot_mutate_content_after_construction() -> None:
    section = PromptSection(
        id="s1", content="original", stability=SectionStability.STABLE
    )
    with pytest.raises(ValidationError):
        section.content = "new content"  # type: ignore[misc]


def test_cannot_mutate_cache_key_after_construction() -> None:
    section = PromptSection(
        id="s1", content="original", stability=SectionStability.STABLE
    )
    with pytest.raises(ValidationError):
        section.cache_key = "forged-key"  # type: ignore[misc]


def test_cache_key_is_derived_from_content() -> None:
    a = PromptSection(
        id="s1", content="alpha", stability=SectionStability.STABLE
    )
    b = PromptSection(
        id="s1", content="beta", stability=SectionStability.STABLE
    )
    assert a.cache_key != b.cache_key


def test_explicit_cache_key_overrides_auto() -> None:
    section = PromptSection(
        id="s1",
        content="anything",
        stability=SectionStability.STABLE,
        cache_key="pinned-by-caller",
    )
    assert section.cache_key == "pinned-by-caller"
