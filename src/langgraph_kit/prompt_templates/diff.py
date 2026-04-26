"""Diff helper for visualizing prompt-section customizations."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_kit.core.prompt_assembly.sections import PromptSection


def diff_section(custom: PromptSection, baseline: PromptSection) -> str:
    """Return a unified diff between *baseline* and *custom* sections.

    Useful as a startup-log entry on deployments that customize the
    shipped prompt-templates library — the actual prompt in use is
    visible without grepping source. The diff covers content,
    version, priority, stability, and condition; ``id`` and
    ``cache_key`` are intentionally omitted (id is the join key,
    cache_key is content-derived).

    Empty string when the two sections are identical (down to
    metadata). The output uses ``difflib.unified_diff`` markers
    (``---`` / ``+++`` / ``@@``) so terminals that highlight diffs
    light up automatically.

    Example::

        from langgraph_kit.prompt_templates import core_identity, diff_section

        custom = core_identity.model_copy(update={
            "content": "You are an internal-tools agent for the X team.",
            "version": "x-team-1",
        })
        print(diff_section(custom, baseline=core_identity))
    """
    if (
        custom.content == baseline.content
        and custom.version == baseline.version
        and custom.priority == baseline.priority
        and custom.stability == baseline.stability
        and custom.condition == baseline.condition
    ):
        return ""

    baseline_lines = _section_lines(baseline)
    custom_lines = _section_lines(custom)
    diff = difflib.unified_diff(
        baseline_lines,
        custom_lines,
        fromfile=f"shipped:{baseline.id}@{baseline.version}",
        tofile=f"custom:{custom.id}@{custom.version}",
        lineterm="",
    )
    return "\n".join(diff)


def _section_lines(section: PromptSection) -> list[str]:
    """Render a section as a stable, line-oriented form for diffing.

    Metadata first, then a blank line, then the content split on
    newlines. ``difflib.unified_diff`` works on iterables of lines;
    keeping the metadata block at the top means version / priority
    bumps land on contiguous lines (easy to read in a diff view).
    """
    return [
        f"# version: {section.version}",
        f"# priority: {section.priority}",
        f"# stability: {section.stability.value}",
        f"# condition: {section.condition or '<none>'}",
        "",
        *section.content.splitlines(),
    ]


__all__ = ["diff_section"]
