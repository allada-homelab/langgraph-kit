"""Regression tests for CoordinatorMode.

The original coordinator sections were marked ``STABLE`` with a
``condition="coordinator"`` attribute that was silently a no-op:
``SectionRegistry.get_active`` only honors ``condition`` on
``CONDITIONAL`` sections. Flipping them to ``CONDITIONAL`` makes the
condition gate work as advertised — these tests lock that in.
"""

from __future__ import annotations

from langgraph_kit.core.coordinator import (
    COORDINATOR_SECTIONS,
    CoordinatorMode,
)
from langgraph_kit.core.prompt_assembly.sections import (
    SectionRegistry,
    SectionStability,
)
from langgraph_kit.core.tools.capability import (
    ToolCapability,
    ToolRisk,
)
from langgraph_kit.core.tools.registry import ToolRegistry


def test_sections_are_conditional_not_stable() -> None:
    """All coordinator sections must be CONDITIONAL so the ``condition``
    attribute actually gates activation."""
    for section in COORDINATOR_SECTIONS:
        assert section.stability == SectionStability.CONDITIONAL, (
            f"Section {section.id!r} marked {section.stability} — condition "
            "gating won't fire. Must be CONDITIONAL."
        )


def test_registry_omits_sections_without_coordinator_condition() -> None:
    registry = SectionRegistry()
    registry.register_many(COORDINATOR_SECTIONS)

    active = registry.get_active(conditions=set())
    coordinator_ids = {s.id for s in COORDINATOR_SECTIONS}
    active_coordinator = [s for s in active if s.id in coordinator_ids]
    assert active_coordinator == [], (
        "Coordinator sections leaked into an assembly with no 'coordinator' "
        f"condition: {[s.id for s in active_coordinator]}"
    )


def test_registry_includes_sections_with_coordinator_condition() -> None:
    registry = SectionRegistry()
    registry.register_many(COORDINATOR_SECTIONS)

    active = registry.get_active(conditions={"coordinator"})
    active_ids = {s.id for s in active}
    for section in COORDINATOR_SECTIONS:
        assert section.id in active_ids, (
            f"Expected {section.id!r} to activate with the 'coordinator' "
            f"condition set; got {active_ids!r}"
        )


def test_get_coordinator_tools_filters_to_read_only() -> None:
    """``get_coordinator_tools`` must not surface mutating or destructive tools.

    ``compile_tools`` returns the raw ``fn`` callables — we match on
    function ``__name__`` to identify which tools came through.
    """

    def read_doc() -> str:
        return "ok"

    def write_doc() -> str:
        return "ok"

    def drop_table() -> str:
        return "boom"

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            id="read_doc",
            name="read_doc",
            description="Read a document",
            fn=read_doc,
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        ToolCapability(
            id="write_doc",
            name="write_doc",
            description="Write a document",
            fn=write_doc,
            risk=ToolRisk.MUTATING,
        )
    )
    registry.register(
        ToolCapability(
            id="drop_table",
            name="drop_table",
            description="Delete everything",
            fn=drop_table,
            risk=ToolRisk.DESTRUCTIVE,
        )
    )

    mode = CoordinatorMode(registry)
    tools = mode.get_coordinator_tools()
    tool_names = {t.__name__ for t in tools}
    assert tool_names == {"read_doc"}, (
        f"Coordinator should only see READ_ONLY tools, got {tool_names!r}"
    )


def test_get_conditions_covers_both_orchestration_keys() -> None:
    conditions = CoordinatorMode.get_conditions()
    assert "coordinator" in conditions
    assert "orchestration" in conditions


def test_coordinator_mode_importable_from_package_root() -> None:
    """Keeping the feature means advertising it at the public surface."""
    from langgraph_kit import COORDINATOR_SECTIONS as root_sections
    from langgraph_kit import CoordinatorMode as RootCoordinatorMode

    assert RootCoordinatorMode is CoordinatorMode
    assert root_sections is COORDINATOR_SECTIONS
