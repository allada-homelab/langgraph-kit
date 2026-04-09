"""Capability activation prompts — teach the agent to discover and use optional features."""

from __future__ import annotations

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

ACTIVATION_SECTIONS: list[PromptSection] = [
    PromptSection(
        id="deferred_tools_awareness",
        content=(
            "# Additional Capabilities\n"
            "Not all tools are loaded by default. If you need a capability "
            "that isn't in your current tool set, use the tool_search tool "
            "to discover additional capabilities. Don't assume a capability "
            "is unavailable — search first."
        ),
        stability=SectionStability.CONDITIONAL,
        priority=55,
        condition="deferred_tools",
    ),
    PromptSection(
        id="skill_activation",
        content=(
            "# Skills\n"
            "Specialized skills are available for specific task types. Each skill "
            "provides a structured workflow with detailed instructions.\n\n"
            "How to use skills:\n"
            "1. Call `list_skills()` to see available skills and their descriptions\n"
            "2. When a task matches a skill, call `read_skill(name)` to load full instructions\n"
            "3. Follow the skill's workflow — skills are structured procedures, not hints\n\n"
            "Only load a skill when you intend to follow its workflow. Do not load "
            "all skills preemptively."
        ),
        stability=SectionStability.CONDITIONAL,
        priority=54,
        condition="skills",
    ),
    PromptSection(
        id="extension_awareness",
        content=(
            "# Extensions\n"
            "Plugin-provided extensions may contribute additional tools, "
            "workers, or prompt guidance. Treat extension-provided capabilities "
            "the same as built-in ones — use them when they're the best fit "
            "for the task."
        ),
        stability=SectionStability.CONDITIONAL,
        priority=53,
        condition="extensions",
    ),
    PromptSection(
        id="async_tasks_awareness",
        content=(
            "# Background Tasks\n"
            "You can launch long-running tasks in the background using "
            "`start_async_task`. After starting a task, return control to the "
            "user — do NOT immediately check the result (this defeats the "
            "purpose of async execution).\n\n"
            "Check tasks with `check_async_task` when:\n"
            "- The user asks about task status\n"
            "- You need the result to proceed with another step\n"
            "- Enough time has passed for the task to complete\n\n"
            "Use `list_async_tasks` to see all tracked tasks and their statuses."
        ),
        stability=SectionStability.CONDITIONAL,
        priority=52,
        condition="async_tasks",
    ),
]
