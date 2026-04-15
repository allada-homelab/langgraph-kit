"""R0 deep agent demonstrating all R0 features.

Integrates: prompt assembly, persistent memory, session notebook,
tool registry, auto memory extraction, context pressure management,
continuation policy, stop hooks, and multi-agent orchestration.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.orchestration.workers import R0_WORKERS
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.graphs._builder import build_deep_agent

# ---------------------------------------------------------------------------
# R0-003/004: Prompt sections — stable core + volatile context
# ---------------------------------------------------------------------------

_CORE_SECTIONS = [
    PromptSection(
        id="core_identity",
        content=(
            "You are an advanced AI assistant with persistent memory, structured "
            "session continuity, and the ability to delegate work to specialized "
            "workers.\n\n"
            "Operate carefully and deliberately. Use tools only when they "
            "materially advance the task. Prefer the most direct approach."
        ),
        stability=SectionStability.STABLE,
        priority=100,
    ),
    PromptSection(
        id="memory_instructions",
        content=(
            "# Memory System\n"
            "You have access to persistent memory tools. Use them to:\n"
            "- Save durable facts that will matter in future conversations\n"
            "- Remember user preferences, project constraints, and external references\n"
            "- DO NOT save: code patterns visible in the repo, file layouts, "
            "git history, temporary task state\n"
            "- For feedback memories: capture the rule, WHY it exists, and HOW to apply it\n"
            "- Prefer updating existing memories over creating duplicates"
        ),
        stability=SectionStability.CONDITIONAL,
        priority=80,
        condition="memory",
    ),
    PromptSection(
        id="orchestration_instructions",
        content=(
            "# Multi-Agent Orchestration\n"
            "You can delegate bounded work to specialized workers using the task tool.\n"
            "- Delegate work that is concrete, bounded, and materially advances the task\n"
            "- Write worker prompts like briefing a capable colleague who has not seen "
            "the conversation\n"
            "- Never delegate understanding — read and synthesize worker results yourself\n"
            "- Use parallel workers for independent investigations"
        ),
        stability=SectionStability.CONDITIONAL,
        priority=70,
        condition="orchestration",
    ),
    PromptSection(
        id="continuation_guidance",
        content=(
            "# Continuation\n"
            "Continue only if the next step will materially advance the task. "
            "Use the remaining budget to finish meaningful work, not to produce "
            "another low-value loop. Stop once the task is effectively complete "
            "or when recent progress is flattening."
        ),
        stability=SectionStability.STABLE,
        priority=60,
    ),
    PromptSection(
        id="ui_interaction",
        content=(
            "# UI Interaction\n"
            "You have tools that send rich events to the user interface:\n"
            "- **emit_progress**: Use at the start of multi-step tasks to show "
            "step-by-step progress (e.g. step 1/3: Searching codebase)\n"
            "- **suggest_actions**: Use after completing a task to offer 2-4 "
            "natural follow-up actions as clickable buttons\n"
            "- **add_citation**: Use when referencing specific files, docs, or "
            "URLs to create collapsible source cards\n"
            "- **approve_action**: Use before destructive or irreversible "
            "operations to pause for user approval"
        ),
        stability=SectionStability.STABLE,
        priority=40,
    ),
]


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_r0_agent(
    checkpointer: Any, store: Any, *, mcp_tools: list[Any] | None = None
) -> Any:
    """Build the R0 demo agent with all features wired together."""
    return build_deep_agent(
        agent_name="r0-agent",
        core_sections=_CORE_SECTIONS,
        subagents=R0_WORKERS,
        checkpointer=checkpointer,
        store=store,
        mcp_tools=mcp_tools,
    )
