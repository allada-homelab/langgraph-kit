"""Memory-awareness sections.

Tells the model that persistent memory exists and frames how to
think about it. The actual memory-extraction prompt
(``langgraph_kit.core.memory.extraction._EXTRACTION_PROMPT``) lives
elsewhere because it's a complete middleware prompt with its own
input/output contract; this section is the chat-side awareness that
"by the way, you have memory tools."

Conditional on ``"memory"`` so it only fires when the agent is built
with memory enabled.
"""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

memory_awareness = PromptSection(
    id="memory_awareness",
    version="1",
    content=(
        "You have access to persistent memory tools. Use them to:\n"
        "- Save durable facts that will matter in future conversations\n"
        "- Remember user preferences, project constraints, and "
        "external references\n"
        "- DO NOT save: code patterns visible in the repo, file "
        "layouts, git history, temporary task state\n"
        "- Prefer updating an existing memory over creating a duplicate"
    ),
    stability=SectionStability.CONDITIONAL,
    condition="memory",
    priority=80,
)
"""Surfaces the memory tool surface to the model.

Mirrors ``langgraph_kit.graphs.reference_deep_agent._CORE_SECTIONS``'s
``memory_instructions`` section but standalone — usable from any agent
build that wires the memory subsystem in. The "DO NOT save" line is
load-bearing: without it, small models fill memory with code-snippet
noise that crowds out genuinely durable facts.
"""


__all__ = ["memory_awareness"]
