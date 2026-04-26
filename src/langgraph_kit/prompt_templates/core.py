"""Core identity + operating-discipline sections.

Captures the "be a careful, concise assistant" framing that almost
every agent built on the kit ends up writing in its system prompt.
Pulled from the same content as
``langgraph_kit.graphs._basic_prompt.BASIC_SYSTEM_PROMPT`` but split
into discrete sections so callers can compose subsets — an
internal-tools agent might want :data:`operate_carefully` and
:data:`be_concise` without :data:`core_identity`'s "AI assistant"
framing, for example.
"""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

core_identity = PromptSection(
    id="core_identity",
    version="1",
    content="You are a helpful AI assistant.",
    stability=SectionStability.STABLE,
    priority=100,
)
"""Establishes the agent as an AI assistant.

Highest-priority STABLE section so it lands at the top of the
composed prompt. Replace via ``model_copy(update={"content": ...})``
when you want a domain-specific persona ("You are a payments-ops
specialist…") instead of the generic framing.
"""


operate_carefully = PromptSection(
    id="operate_carefully",
    version="1",
    content=(
        "Operate carefully and deliberately. When tools are available, "
        "use them only when they materially advance the task. Prefer "
        "the most direct approach."
    ),
    stability=SectionStability.STABLE,
    priority=90,
)
"""Operating-discipline guidance.

Discourages tool spam and over-engineered approaches. Pairs well
with :data:`core_identity` and :data:`be_concise` for a baseline
"thoughtful agent" prompt.
"""


be_concise = PromptSection(
    id="be_concise",
    version="1",
    content=(
        "Be concise. State results and decisions directly. If you "
        "cannot answer or complete something, say so explicitly "
        "rather than guessing."
    ),
    stability=SectionStability.STABLE,
    priority=80,
)
"""Pushes terse, honest responses.

The "say so explicitly rather than guessing" clause is load-bearing
— without it, instruction-following models tend to invent answers
when they don't know something.
"""


completion_signal = PromptSection(
    id="completion_signal",
    version="1",
    content=(
        "When the task is complete, end your response with a clear "
        "signal that the work is done — don't trail off mid-thought. "
        "If you're blocked or need input, say so explicitly so the "
        "user knows it's their turn."
    ),
    stability=SectionStability.STABLE,
    priority=60,
)
"""Encourages clear turn-handoff signaling.

Helps with multi-turn agents and agents wired into HITL where the
caller needs to know whether the agent is genuinely done or
expecting a follow-up. Lower priority than identity/discipline
because it's a polish on top of those.
"""


__all__ = [
    "be_concise",
    "completion_signal",
    "core_identity",
    "operate_carefully",
]
