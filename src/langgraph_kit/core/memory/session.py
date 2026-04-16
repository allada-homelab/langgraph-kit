"""Thread-local session notebook for maintaining continuity within a conversation."""

from __future__ import annotations

from typing import Any

NOTEBOOK_SECTIONS = [
    "Current State",
    "Task Specification",
    "Files and Functions",
    "Workflow",
    "Errors and Corrections",
    "Key Results",
    "Worklog",
]

NOTEBOOK_TEMPLATE = "# Session Notebook\n\n" + "\n\n".join(
    f"## {section}\n" for section in NOTEBOOK_SECTIONS
)

# Update thresholds
DEFAULT_MESSAGE_THRESHOLD = 6
DEFAULT_TOOL_CALL_THRESHOLD = 4
DEFAULT_MAX_SECTION_TOKENS = 500  # Approximate, per section
DEFAULT_MAX_TOTAL_TOKENS = 3000  # Approximate, total notebook


def _find_section_bounds(text: str, section: str) -> tuple[int, int] | None:
    """Find (start, end) character positions of a section's content in the notebook."""
    header = f"## {section}\n"
    if header not in text:
        return None
    start = text.index(header) + len(header)
    end = len(text)
    for s in NOTEBOOK_SECTIONS:
        marker = f"## {s}\n"
        pos = text.find(marker, start)
        if pos != -1 and pos < end:
            end = pos
    return start, end


def _condense_section_in_text(text: str, section: str, max_tokens: int) -> str:
    """Truncate a section in-memory if it exceeds max_tokens estimate."""
    bounds = _find_section_bounds(text, section)
    if bounds is None:
        return text
    start, end = bounds
    content = text[start:end].strip()
    if len(content) // 4 <= max_tokens:
        return text
    max_chars = max_tokens * 4
    truncated = "...\n" + content[-max_chars:]
    return text[:start] + truncated + "\n\n" + text[end:]


class SessionNotebook:
    """Structured notebook stored per-thread in LangGraph Store.

    The notebook is persisted under namespace ``("session", thread_id)`` with a
    single key ``"notebook"``.  It provides helpers for loading, saving, and
    selectively updating individual sections while preserving the overall
    Markdown structure.
    """

    def __init__(self, store: Any, thread_id: str) -> None:
        super().__init__()
        self._store = store
        self._thread_id = thread_id
        self._namespace = ("session", thread_id)
        self._key = "notebook"

    async def initialize(self) -> None:
        """Create notebook from template if it doesn't already exist."""
        existing = await self._store.aget(self._namespace, self._key)
        if existing is None:
            await self._store.aput(
                self._namespace, self._key, {"content": NOTEBOOK_TEMPLATE}
            )

    async def load(self) -> str:
        """Load and return the current notebook content."""
        item = await self._store.aget(self._namespace, self._key)
        if item is None:
            await self.initialize()
            return NOTEBOOK_TEMPLATE
        return item.value.get("content", NOTEBOOK_TEMPLATE)

    async def save(self, content: str) -> None:
        """Overwrite the entire notebook content."""
        await self._store.aput(self._namespace, self._key, {"content": content})

    async def update_section(self, section: str, content: str) -> None:
        """Replace content of a specific section, preserving notebook structure."""
        notebook = await self.load()
        bounds = _find_section_bounds(notebook, section)
        if bounds is None:
            return
        start, end = bounds
        updated = notebook[:start] + content.strip() + "\n\n" + notebook[end:]
        await self.save(updated.strip() + "\n")

    async def get_section(self, section: str) -> str:
        """Extract the content of a specific section."""
        notebook = await self.load()
        bounds = _find_section_bounds(notebook, section)
        if bounds is None:
            return ""
        start, end = bounds
        return notebook[start:end].strip()

    def should_update(self, messages_since: int, tool_calls_since: int) -> bool:
        """Decide if notebook should update based on activity thresholds."""
        return (
            messages_since >= DEFAULT_MESSAGE_THRESHOLD
            or tool_calls_since >= DEFAULT_TOOL_CALL_THRESHOLD
        )

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token heuristic)."""
        return len(text) // 4

    async def get_token_estimate(self) -> int:
        """Estimate total tokens in current notebook."""
        content = await self.load()
        return self.estimate_tokens(content)

    async def condense_section(self, section: str, max_tokens: int) -> None:
        """Truncate a section if it exceeds max_tokens estimate."""
        content = await self.get_section(section)
        if self.estimate_tokens(content) <= max_tokens:
            return
        max_chars = max_tokens * 4
        truncated = "...\n" + content[-max_chars:]
        await self.update_section(section, truncated)

    async def enforce_budget(self) -> None:
        """Condense sections if total notebook exceeds budget.

        Performs a single load, all truncations in memory, then a single save.
        """
        notebook = await self.load()
        if self.estimate_tokens(notebook) <= DEFAULT_MAX_TOTAL_TOKENS:
            return

        for section in NOTEBOOK_SECTIONS:
            notebook = _condense_section_in_text(
                notebook, section, DEFAULT_MAX_SECTION_TOKENS
            )
        await self.save(notebook.strip() + "\n")
