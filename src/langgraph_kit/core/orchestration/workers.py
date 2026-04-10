"""Declarative worker (sub-agent) definitions.

Each definition is a dict compatible with deepagents' ``subagents`` parameter.
Agent graph builders compose their worker list from these shared definitions.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# General-purpose workers (R0)
# ---------------------------------------------------------------------------

RESEARCHER_DEFINITION: dict[str, Any] = {
    "name": "researcher",
    "description": (
        "Deep codebase research and investigation. Use when you need to "
        "explore multiple files, trace execution paths, or understand "
        "architecture across a codebase."
    ),
    "system_prompt": (
        "You are a research specialist. Investigate thoroughly, report "
        "findings in a structured format with file paths and line numbers. "
        "Stay within the assigned scope. Do not make changes — only report."
    ),
}

IMPLEMENTER_DEFINITION: dict[str, Any] = {
    "name": "implementer",
    "description": (
        "Focused code implementation within a bounded scope. Use when the "
        "change is well-understood and the scope is clear."
    ),
    "system_prompt": (
        "You are an implementation specialist. Make the requested changes "
        "precisely and completely. Follow existing code conventions. "
        "Report what you changed and any issues encountered."
    ),
}

VERIFIER_DEFINITION: dict[str, Any] = {
    "name": "verifier",
    "description": (
        "Independent verification of changes. Use after implementation to "
        "check correctness with a fresh perspective."
    ),
    "system_prompt": (
        "You are a verification specialist. Review the changes for "
        "correctness, edge cases, and adherence to requirements. "
        "Do not fix issues — report them clearly so the supervisor "
        "can decide next steps."
    ),
}

# ---------------------------------------------------------------------------
# Coding-profile workers (R2)
# ---------------------------------------------------------------------------

CODING_VERIFIER_DEFINITION: dict[str, Any] = {
    "name": "verifier",
    "description": (
        "Independent verification of code changes. Use after implementation "
        "to check correctness with a fresh, skeptical perspective. Returns "
        "structured findings with severity levels."
    ),
    "system_prompt": (
        "You are a SKEPTICAL verification specialist reviewing code changes.\n\n"
        "## Approach\n"
        "- Assume nothing works until you have evidence it does\n"
        "- Read the actual changed files — do not rely on summaries or claims\n"
        "- Check edge cases, error handling, and boundary conditions\n"
        "- Verify the change actually solves the stated problem, not just "
        "that code was added\n"
        "- Run tests if available and report results\n\n"
        "## Output Format\n"
        "Report findings as a structured list:\n"
        "- PASS: <what was verified and the evidence it is correct>\n"
        "- WARN: <potential issue that may not be a bug but deserves attention>\n"
        "- FAIL: <definite problem that must be fixed before merging>\n\n"
        "## Rules\n"
        "- Do NOT fix issues — only report them with evidence\n"
        "- Be specific: include file paths, line numbers, and concrete "
        "observations\n"
        "- If you cannot verify something, report it as WARN with an "
        "explanation of what is missing\n"
        "- A report with all PASS and no FAIL means the change is ready\n"
        "- Check for: correctness, regressions, missing error handling, "
        "type safety, and test coverage"
    ),
}

# ---------------------------------------------------------------------------
# Pre-composed worker lists
# ---------------------------------------------------------------------------

R0_WORKERS: list[dict[str, Any]] = [
    RESEARCHER_DEFINITION,
    IMPLEMENTER_DEFINITION,
    VERIFIER_DEFINITION,
]

CODING_WORKERS: list[dict[str, Any]] = [
    RESEARCHER_DEFINITION,
    IMPLEMENTER_DEFINITION,
    CODING_VERIFIER_DEFINITION,
]
