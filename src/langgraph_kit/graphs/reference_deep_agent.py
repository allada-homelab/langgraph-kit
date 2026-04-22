"""Reference full-stack deep agent.

Wires every kit feature together: prompt assembly, persistent memory,
session notebook, tool registry, auto memory extraction, context pressure
management (including full compaction), continuation policy, stop hooks,
and multi-agent orchestration. Clone this module when starting a new
domain-specific agent and layer on domain-specific sections, providers,
and tools (see ``coding_agent`` for the canonical extension pattern).
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.graphs._builder import build_deep_agent

# ---------------------------------------------------------------------------
# Prompt sections — stable core + volatile context
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
            "After each step, ask: did this step produce new, useful output? "
            "If two consecutive steps produced no meaningful progress (no new "
            "information, no code changes, no resolved errors), stop and report "
            "current state. Do not loop on the same error more than twice without "
            "changing approach."
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
            "operations to pause for user approval\n\n"
            "Skip emit_progress for tasks that complete in a single tool call. "
            "Skip suggest_actions when the user's intent is already clear."
        ),
        stability=SectionStability.STABLE,
        priority=40,
    ),
]


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_reference_deep_agent(
    checkpointer: Any, store: Any, *, mcp_tools: list[Any] | None = None
) -> Any:
    """Build the reference deep agent with all kit features wired together."""
    return build_deep_agent(
        agent_name="reference-deep-agent",
        core_sections=_CORE_SECTIONS,
        subagents=GENERAL_WORKERS,
        checkpointer=checkpointer,
        store=store,
        mcp_tools=mcp_tools,
    )
