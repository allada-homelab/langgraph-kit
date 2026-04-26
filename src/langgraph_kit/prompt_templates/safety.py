"""Error-handling and output-format sections."""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

error_handling = PromptSection(
    id="error_handling",
    version="1",
    content=(
        "When something goes wrong:\n"
        "- Read the full error message before deciding what to do\n"
        "- Don't keep retrying the same approach if it's already failed twice\n"
        "- If you're stuck, surface what you tried and what blocked you "
        "rather than silently producing a degraded answer"
    ),
    stability=SectionStability.STABLE,
    priority=70,
)
"""Error-handling discipline.

Counters two common LLM failure modes: skimming errors (so the model
misses the part that says "you must include field X") and infinite
retry loops on the same approach. The "surface what blocked you"
clause is critical for HITL flows where a human is waiting to help.
"""


output_format_natural = PromptSection(
    id="output_format",
    version="1-natural",
    content=(
        "Reply in plain natural language. No markdown headers unless "
        "the user's question genuinely benefits from structure. No "
        'preamble like "Sure!" or "Here\'s the answer:" — just '
        "the answer."
    ),
    stability=SectionStability.STABLE,
    priority=50,
)
"""Plain natural-language output formatting.

Pairs with chat-style consumers where markdown structure is noise.
Note the ``id="output_format"`` is shared with
:data:`output_format_structured` — they're meant to be mutually
exclusive (callers register one or the other), not both. The
``version`` field disambiguates which variant landed in a given
:class:`SectionRegistry`.
"""


output_format_structured = PromptSection(
    id="output_format",
    version="1-structured",
    content=(
        "Use markdown structure when it helps the user scan the "
        "response: section headers for distinct topics, bullet "
        "lists for enumerations, code fences for code or shell "
        "commands. Don't add structure for the sake of it — a "
        "single-paragraph answer is a single paragraph."
    ),
    stability=SectionStability.STABLE,
    priority=50,
)
"""Markdown-structured output formatting.

Mirror of :data:`output_format_natural` for callers whose UI renders
markdown well (a docs viewer, a developer console, etc.). Same id,
different version — :class:`SectionRegistry`'s versioning support
(issue #18) lets a single registry hold both and switch via
``set_current``.
"""


__all__ = [
    "error_handling",
    "output_format_natural",
    "output_format_structured",
]
