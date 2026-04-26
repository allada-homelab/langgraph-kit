"""Multi-agent / coordinator orchestration sections.

Curated counterparts to the ad-hoc orchestration prompts that ship
inline today (``CoordinatorMode`` in
``langgraph_kit.core.coordinator``, the
``orchestration_instructions`` section in
``reference_deep_agent``). Same shape as the rest of the
``prompt_templates`` library — ``CONDITIONAL`` sections gated on
``"orchestration"`` so they activate only when the build registers
the condition.

Use these when assembling a custom agent that should delegate work
or coordinate sub-workers but doesn't otherwise want to inherit
``CoordinatorMode``'s strict read-only-tool surface.
"""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

delegation_discipline = PromptSection(
    id="delegation_discipline",
    version="1",
    content=(
        "# Delegation Discipline\n"
        "When you delegate work to a sub-worker:\n"
        "- Brief them like a colleague who hasn't seen the conversation.\n"
        "- Include: the goal, relevant context, what success looks like.\n"
        "- Keep delegations bounded — concrete tasks with clear exit conditions.\n"
        "- Never delegate understanding — read worker results yourself first."
    ),
    stability=SectionStability.CONDITIONAL,
    priority=80,
    condition="orchestration",
)
"""Briefing-quality discipline for delegations.

Drop in when the agent has a ``task`` (or equivalent) tool and you
want it to write good worker prompts instead of one-line dispatches.
"""

synthesis_discipline = PromptSection(
    id="synthesis_discipline",
    version="1",
    content=(
        "# Synthesis Discipline\n"
        "When workers return results:\n"
        "1. Read and understand the findings yourself.\n"
        "2. Identify gaps, contradictions, or next steps.\n"
        "3. Synthesize into a clear answer for the user.\n"
        "Do NOT relay worker output verbatim. Add value through integration."
    ),
    stability=SectionStability.CONDITIONAL,
    priority=78,
    condition="orchestration",
)
"""Forces the agent to integrate worker output rather than echo it.

Tracks the same intent as ``CoordinatorMode``'s synthesis section
but lives in the curated library so non-coordinator agents can
adopt it without inheriting the read-only tool filter.
"""

parallel_workers = PromptSection(
    id="parallel_workers",
    version="1",
    content=(
        "# Parallel Workers\n"
        "When delegations are independent (no data flowing between them), "
        "fire them in parallel rather than sequentially. Sequential delegation "
        "is appropriate when worker B depends on worker A's findings."
    ),
    stability=SectionStability.CONDITIONAL,
    priority=70,
    condition="orchestration",
)
"""Pushes the agent to parallelise independent delegations.

Pairs with the kit's ``task`` tool which accepts a list of
delegations.
"""


__all__ = [
    "delegation_discipline",
    "parallel_workers",
    "synthesis_discipline",
]
