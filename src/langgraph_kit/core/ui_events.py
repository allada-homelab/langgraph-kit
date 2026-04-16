"""UI event tools — emit rich frontend events alongside text tokens.

Agents call these tools to produce ephemeral UI elements (progress bars,
suggested actions, citations) that the frontend renders inline in the chat.
Unlike artifacts, these are transient and don't persist in a side panel.

Each tool returns a sentinel-prefixed JSON string that the streaming layer
detects in ``on_tool_end`` and emits as a dedicated SSE event.
"""

from __future__ import annotations

import json
from typing import Any

# Sentinel prefixes detected by streaming.py
PROGRESS_SENTINEL = "__progress__:"
SUGGESTIONS_SENTINEL = "__suggestions__:"
CITATION_SENTINEL = "__citation__:"

MAX_SUGGESTIONS = 6
MAX_ACTION_LABEL_LEN = 80


def build_progress_tool() -> Any:
    """Create the ``emit_progress`` tool for agents."""

    async def emit_progress(
        step: str,
        current: int,
        total: int,
    ) -> str:
        """Emit a progress update to the user interface.

        Use this to show the user where you are in a multi-step process.
        The frontend renders this as a progress bar with step description.

        Args:
            step: Description of the current step (e.g. "Searching codebase")
            current: Current step number (1-based)
            total: Total number of steps
        """
        if total < 1:
            return "Error: total must be >= 1"
        if current < 1:
            return "Error: current must be >= 1 (progress is 1-based)"
        if current > total:
            return f"Error: current ({current}) cannot exceed total ({total})"
        step = step.strip()
        if not step:
            return "Error: step description must not be empty"

        payload = {"step": step, "current": current, "total": total}
        return PROGRESS_SENTINEL + json.dumps(payload)

    return emit_progress


def build_suggestions_tool() -> Any:
    """Create the ``suggest_actions`` tool for agents."""

    async def suggest_actions(
        actions: list[str],
    ) -> str:
        """Suggest follow-up actions the user can take.

        The frontend renders these as clickable buttons below your response.
        Use this at the end of a task to offer natural next steps.

        Args:
            actions: List of short action labels (e.g. ["Run tests", "Review changes", "Deploy"])
        """
        cleaned = [a.strip() for a in actions if isinstance(a, str) and a.strip()]
        if not cleaned:
            return "Error: actions must contain at least one non-empty label"
        if len(cleaned) > MAX_SUGGESTIONS:
            return f"Error: cannot suggest more than {MAX_SUGGESTIONS} actions (got {len(cleaned)})"
        truncated = [
            a if len(a) <= MAX_ACTION_LABEL_LEN else a[: MAX_ACTION_LABEL_LEN - 1] + "…"
            for a in cleaned
        ]
        payload = {"actions": truncated}
        return SUGGESTIONS_SENTINEL + json.dumps(payload)

    return suggest_actions


def build_citation_tool() -> Any:
    """Create the ``add_citation`` tool for agents."""

    async def add_citation(
        title: str,
        source: str,
        snippet: str = "",
    ) -> str:
        """Add a citation or source reference to your response.

        The frontend renders this as a collapsible source card.
        Use this when referencing files, documentation, or external sources.

        Args:
            title: Short title for the source (e.g. "auth.py:42")
            source: URL or file path of the source
            snippet: Optional relevant excerpt from the source
        """
        title = title.strip()
        source = source.strip()
        if not title:
            return "Error: title must not be empty"
        if not source:
            return "Error: source must not be empty"
        payload = {"title": title, "source": source, "snippet": snippet}
        return CITATION_SENTINEL + json.dumps(payload)

    return add_citation
