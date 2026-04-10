"""Coding agent — R0 harness with R2 coding-profile overlays.

Composes the general-purpose R0 infrastructure (middleware, memory, tools)
with coding-specific prompt sections, a git context provider, worktree
tools, an enhanced verification worker, and slash-command dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.orchestration.workers import CODING_WORKERS
from langgraph_kit.core.prompt_assembly.activation import ACTIVATION_SECTIONS
from langgraph_kit.core.prompt_assembly.coding_sections import (
    CODING_SEARCH_SECTIONS,
    CODING_WORKFLOW_SECTIONS,
)
from langgraph_kit.core.prompt_assembly.composer import PromptComposer
from langgraph_kit.core.prompt_assembly.context_providers import (
    MemoryContextProvider,
    ThreadContextProvider,
    ToolContextProvider,
)
from langgraph_kit.core.prompt_assembly.git_context import GitContextProvider
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionRegistry,
    SectionStability,
)
from langgraph_kit.core.graph_builder.backend import build_backend_factory
from langgraph_kit.core.graph_builder.commands import build_command_dispatcher
from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
from langgraph_kit.core.graph_builder.tools import register_standard_tools, register_tool
from langgraph_kit.core.tools.capability import ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.core.tools.worktree import (
    WORKTREE_GUIDANCE_SECTION,
    build_worktree_tools,
)
from langgraph_kit.graphs.r0_agent import _CORE_SECTIONS
from langgraph_kit.llm import build_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_coding_agent(
    checkpointer: Any, store: Any, *, mcp_tools: list[Any] | None = None
) -> Any:
    """Build the coding agent with R0 infrastructure + R2 coding overlays."""
    from deepagents import (
        create_deep_agent,  # pyright: ignore[reportMissingModuleSource]
    )

    llm = build_llm()
    memory_mgr = PersistentMemoryManager(store)

    # --- Tool registry: standard suite + coding-specific worktree tools ---
    tool_registry = ToolRegistry()
    register_standard_tools(
        tool_registry,
        memory_mgr,
        store,
        parent_thread_id="coding-global",
        mcp_tools=mcp_tools,
    )

    # R2-004: Worktree tools (coding-specific)
    for i, tool_fn in enumerate(build_worktree_tools()):
        name = getattr(tool_fn, "__name__", f"worktree_tool_{i}")
        register_tool(
            tool_registry,
            tool_fn,
            id_prefix="worktree",
            tags=["git", "worktree"],
            risk=ToolRisk.READ_ONLY if name == "list_worktrees" else ToolRisk.MUTATING,
        )

    # --- Prompt assembly: R0 core + coding overlays ---
    section_registry = SectionRegistry()
    section_registry.register_many(_CORE_SECTIONS)
    section_registry.register_many(ACTIVATION_SECTIONS)
    section_registry.register_many(CODING_WORKFLOW_SECTIONS)
    section_registry.register_many(CODING_SEARCH_SECTIONS)
    section_registry.register(WORKTREE_GUIDANCE_SECTION)

    tool_guidance = tool_registry.collect_prompt_fragments()
    if tool_guidance:
        section_registry.register(
            PromptSection(
                id="tool_guidance",
                content=tool_guidance,
                stability=SectionStability.VOLATILE,
                priority=50,
            )
        )

    providers = [
        ThreadContextProvider(),
        MemoryContextProvider(),
        ToolContextProvider(),
        GitContextProvider(),  # R2-002: Git context
    ]
    composer = PromptComposer(section_registry, providers)

    # --- Commands + middleware ---
    pressure_monitor = PressureMonitor()
    command_dispatcher = build_command_dispatcher(
        memory_mgr, pressure_monitor, tool_registry=tool_registry
    )
    middleware, _ = build_middleware_stack(
        llm=llm,
        memory_mgr=memory_mgr,
        pressure_monitor=pressure_monitor,
        command_dispatcher=command_dispatcher,
    )

    # --- Compose system prompt ---
    system_prompt = composer.compose_sections_only(
        conditions={
            "memory",
            "orchestration",
            "deferred_tools",
            "skills",
            "async_tasks",
        }
    )

    # --- Build the deep agent ---
    graph = create_deep_agent(
        model=llm,
        tools=tool_registry.compile_tools(),
        system_prompt=system_prompt,
        middleware=middleware,
        subagents=CODING_WORKERS,
        checkpointer=checkpointer,
        store=store,
        backend=build_backend_factory("coding_agent"),
        name="coding-agent",
    )
    return graph, command_dispatcher
