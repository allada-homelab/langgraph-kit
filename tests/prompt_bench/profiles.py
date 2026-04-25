"""Profile bridges — translate an overlay into a built agent graph.

Each profile knows:

- Which agent module to build.
- Where the agent's section list lives (so an overlay's ``section_overrides``
  can be applied to a fresh copy without monkey-patching module state).
- How to compile the graph end-to-end.

This module is the *only* place that knows about specific agent
profiles. The rest of the bench harness stays generic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph_kit.core.prompt_assembly.sections import PromptSection
from tests.prompt_bench.variants import PromptOverlay


@dataclass(frozen=True)
class ProfileSpec:
    """Description of an agent profile the bench can build."""

    name: str
    section_source_module: str  # e.g. "langgraph_kit.graphs.reference_deep_agent"
    section_attr: str  # e.g. "_CORE_SECTIONS"


PROFILES: dict[str, ProfileSpec] = {
    "reference_deep_agent": ProfileSpec(
        name="reference_deep_agent",
        section_source_module="langgraph_kit.graphs.reference_deep_agent",
        section_attr="_CORE_SECTIONS",
    ),
    "coding_agent": ProfileSpec(
        name="coding_agent",
        section_source_module="langgraph_kit.graphs.coding_agent",
        section_attr="_CORE_SECTIONS",
    ),
}


def get_baseline_sections(profile_name: str) -> list[PromptSection]:
    """Return a fresh copy of the baseline section list for *profile_name*."""
    spec = _require_profile(profile_name)
    import importlib

    module = importlib.import_module(spec.section_source_module)
    sections = getattr(module, spec.section_attr, None)
    if not isinstance(sections, list):
        msg = (
            f"Profile {profile_name!r}: expected list at "
            f"{spec.section_source_module}.{spec.section_attr}, got "
            f"{type(sections).__name__}"
        )
        raise RuntimeError(msg)
    return list(sections)


def apply_overlay_to_sections(
    profile_name: str, overlay: PromptOverlay
) -> list[PromptSection]:
    """Return the profile's section list with *overlay* applied.

    Section IDs in ``overlay.section_overrides`` that don't exist in
    the profile's list are an error — overlays must target existing
    sections so we know exactly what we're swapping.
    """
    sections = get_baseline_sections(profile_name)
    if not overlay.section_overrides:
        return sections

    by_id = {s.id: s for s in sections}
    missing = set(overlay.section_overrides) - by_id.keys()
    if missing:
        msg = (
            f"Profile {profile_name!r} has no sections named "
            f"{sorted(missing)} (cannot apply overlay {overlay.name!r})"
        )
        raise KeyError(msg)

    swapped: list[PromptSection] = []
    for section in sections:
        if section.id in overlay.section_overrides:
            swapped.append(
                PromptSection(
                    id=section.id,
                    content=overlay.section_overrides[section.id],
                    stability=section.stability,
                    priority=section.priority,
                    condition=section.condition,
                )
            )
        else:
            swapped.append(section)
    return swapped


# ---------------------------------------------------------------------------
# Graph builders — keyed by profile name. Tests can register additional
# builders for hermetic profiles via :func:`register_profile_builder`.
# ---------------------------------------------------------------------------

ProfileBuilder = Callable[[PromptOverlay, Any], Any]
"""``(overlay, deps) -> compiled_graph``. ``deps`` is profile-specific."""

_BUILDERS: dict[str, ProfileBuilder] = {}


def register_profile_builder(name: str, builder: ProfileBuilder) -> None:
    """Register a graph builder for *name*. Used by tests to inject mocks."""
    _BUILDERS[name] = builder


def build_profile_graph(profile_name: str, overlay: PromptOverlay, deps: Any) -> Any:
    """Build the graph for *profile_name* with *overlay* applied."""
    if profile_name not in _BUILDERS:
        msg = (
            f"No graph builder registered for profile {profile_name!r}. "
            f"Available: {sorted(_BUILDERS)}"
        )
        raise KeyError(msg)
    return _BUILDERS[profile_name](overlay, deps)


# ---------------------------------------------------------------------------
# Default reference_deep_agent builder — wired up lazily so importing this
# module doesn't pay LangGraph's import cost when tests only need the
# overlay-application logic.
# ---------------------------------------------------------------------------


def install_default_builders() -> None:
    """Register builders for the agent profiles shipped with the kit.

    Idempotent — safe to call from multiple fixtures. Each builder
    expects ``deps`` to be a ``(checkpointer, store)`` tuple, matching
    the kit's ``build_*_agent`` signatures.
    """
    if "reference_deep_agent" not in _BUILDERS:
        register_profile_builder("reference_deep_agent", _build_reference_deep_agent)
    if "coding_agent" not in _BUILDERS:
        register_profile_builder("coding_agent", _build_coding_agent)


def _build_reference_deep_agent(overlay: PromptOverlay, deps: Any) -> Any:
    """Build the reference deep agent with section-overlay applied.

    Note: middleware overlays (e.g. ``_EXTRACTION_PROMPT``) are NOT
    applied here — the caller (``live_runner.run_one``) keeps
    ``patch_middleware_prompts`` active across both build and invoke,
    since middleware constants are read each turn.
    """
    from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
    from langgraph_kit.graphs._builder import build_deep_agent

    sections = apply_overlay_to_sections("reference_deep_agent", overlay)
    checkpointer, store = deps
    # ``build_deep_agent`` returns ``(graph, command_dispatcher)``; the
    # bench only needs the graph, so unpack and discard the dispatcher
    # (the e2e suite + ``examples/reference_deep_agent.py`` follow the
    # same pattern).
    graph, _dispatcher = build_deep_agent(
        agent_name="reference-deep-agent",
        core_sections=sections,
        subagents=GENERAL_WORKERS,
        checkpointer=checkpointer,
        store=store,
    )
    return graph


def _build_coding_agent(overlay: PromptOverlay, deps: Any) -> Any:
    """Build the coding agent with section-overlay applied.

    See ``_build_reference_deep_agent`` for the middleware-patch
    responsibility split.
    """
    from langgraph_kit.core.orchestration.workers import CODING_WORKERS
    from langgraph_kit.graphs._builder import build_deep_agent

    sections = apply_overlay_to_sections("coding_agent", overlay)
    checkpointer, store = deps
    graph, _dispatcher = build_deep_agent(
        agent_name="coding-agent",
        core_sections=sections,
        subagents=CODING_WORKERS,
        checkpointer=checkpointer,
        store=store,
    )
    return graph


def _require_profile(name: str) -> ProfileSpec:
    if name not in PROFILES:
        msg = f"Unknown agent profile {name!r}. Available: {sorted(PROFILES)}"
        raise KeyError(msg)
    return PROFILES[name]
