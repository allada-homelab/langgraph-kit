"""Tool-use guidance sections."""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

tool_use_discipline = PromptSection(
    id="tool_use_discipline",
    version="1",
    content=(
        "When you have multiple tool calls to make and they're "
        "independent of each other, batch them in parallel rather "
        "than serializing — saves wall time and is the user's "
        "common expectation. If a tool call fails, read the error "
        "and fix the cause; don't retry blindly more than once."
    ),
    stability=SectionStability.CONDITIONAL,
    condition="tools",
    priority=85,
)
"""Tool-use discipline: parallelize independent calls, don't retry blindly.

Conditional on ``"tools"`` because non-tool agents shouldn't see
this. The "fix the cause; don't retry blindly" framing addresses
the most expensive tool-loop failure mode (model retries the same
call expecting different output). Pairs with the kit's
``ToolErrorMiddleware``, which surfaces the actual error text the
model needs to act on.
"""


__all__ = ["tool_use_discipline"]
