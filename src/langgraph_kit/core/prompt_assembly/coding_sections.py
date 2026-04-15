"""Coding-profile prompt overlay sections (R2-001, R2-003).

These sections add repository-aware work habits, verification expectations,
and search-first discipline to the coding agent. They are additive overlays —
they do not modify any core or activation section.
"""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

# ---------------------------------------------------------------------------
# R2-001: Coding Workflow Prompt Rules
# ---------------------------------------------------------------------------

CODING_WORKFLOW_SECTIONS = [
    PromptSection(
        id="coding_workflow_rules",
        content=(
            "# Coding Workflow\n"
            "When working with code, follow these rules:\n"
            "1. **Read before edit** — Always read the relevant code before making "
            "changes. Understand the current state, surrounding context, and "
            "conventions before modifying anything.\n"
            "2. **Targeted changes** — Each edit should change only what is necessary "
            "to accomplish the task. Do not rewrite entire files or refactor "
            "unrelated code.\n"
            "3. **Preserve unrelated work** — Do not alter imports, formatting, "
            "comments, or logic that is outside the scope of the current change. "
            "Respect in-progress work on other branches.\n"
            "4. **Verify before done** — After making changes, read the modified "
            "files to confirm correctness. Run tests if available. Do not assume "
            "edits were applied correctly."
        ),
        stability=SectionStability.STABLE,
        priority=38,
    ),
]

# ---------------------------------------------------------------------------
# R2-003: File Editing and Search-First Workflow
# ---------------------------------------------------------------------------

CODING_SEARCH_SECTIONS = [
    PromptSection(
        id="coding_search_first",
        content=(
            "# Search-First File Workflow\n"
            "1. **Search before assumptions** — Use search and grep tools to find "
            "the relevant code path before changing anything. Do not guess file "
            "locations or function signatures.\n"
            "2. **Read the target region** — Before editing a file, read the exact "
            "region you plan to change so your edit accounts for surrounding "
            "context.\n"
            "3. **Targeted diffs** — Prefer surgical, line-level edits over "
            "full-file rewrites. A precise replacement is safer and easier to "
            "review than regenerating an entire file.\n"
            "4. **Minimal file set** — Touch only the files directly required by "
            "the task. Fewer changed files means fewer opportunities for "
            "unintended side effects."
        ),
        stability=SectionStability.STABLE,
        priority=35,
    ),
]
