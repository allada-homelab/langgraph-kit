"""Command dispatcher assembly for agent graph builders."""

from __future__ import annotations

from langgraph_kit.core.commands.builtins import (
    build_compact_command,
    build_context_command,
    build_help_command,
    build_memory_command,
    build_skills_command,
    build_status_command,
    build_tools_command,
)
from langgraph_kit.core.commands.dispatch import CommandDispatcher
from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.tools.registry import ToolRegistry


def build_command_dispatcher(
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    tool_registry: ToolRegistry | None = None,
) -> CommandDispatcher:
    """Build the standard slash-command dispatcher."""
    from langgraph_kit.core.skills.registry import SkillRegistry as _SkillRegistry

    dispatcher = CommandDispatcher()
    dispatcher.register(
        "help",
        build_help_command(dispatcher),
        description="List available commands",
    )
    dispatcher.register(
        "memory",
        build_memory_command(memory_mgr),
        description="Inspect stored memories",
        usage="[scope]",
    )
    dispatcher.register(
        "context",
        build_context_command(pressure_monitor),
        description="Show context window status",
    )
    dispatcher.register(
        "compact",
        build_compact_command(pressure_monitor),
        description="Truncate large tool outputs to free context space",
    )
    dispatcher.register(
        "status",
        build_status_command(pressure_monitor, memory_mgr),
        description="Combined dashboard: context, memory, status",
    )
    if tool_registry is not None:
        dispatcher.register(
            "tools",
            build_tools_command(tool_registry),
            description="List registered tools",
            usage="[tag]",
        )
    # Skills are always available (loaded from disk)
    dispatcher.register(
        "skills",
        build_skills_command(_SkillRegistry()),
        description="List available skills",
    )
    return dispatcher
