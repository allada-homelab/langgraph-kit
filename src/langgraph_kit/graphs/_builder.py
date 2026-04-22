"""Shared agent builder — extracts the common skeleton for deep agent construction.

Both r0_agent and coding_agent follow the same build sequence (LLM → tools →
prompt assembly → middleware → create_deep_agent). This module captures that
skeleton so each agent only specifies its unique overlays.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.graph_builder.backend import build_backend_factory
from langgraph_kit.core.graph_builder.commands import build_command_dispatcher
from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
from langgraph_kit.core.graph_builder.tools import register_standard_tools
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.prompt_assembly.activation import ACTIVATION_SECTIONS
from langgraph_kit.core.prompt_assembly.composer import PromptComposer
from langgraph_kit.core.prompt_assembly.context_providers import (
    MemoryContextProvider,
    ThreadContextProvider,
    ToolContextProvider,
)
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionRegistry,
    SectionStability,
)
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.llm import build_llm


def build_deep_agent(
    *,
    agent_name: str,
    core_sections: list[PromptSection],
    subagents: list[dict[str, Any]],
    checkpointer: Any,
    store: Any,
    mcp_tools: list[Any] | None = None,
    extra_sections: list[list[PromptSection]] | None = None,
    extra_providers: list[Any] | None = None,
    configure_tools: Any | None = None,
    conditions: set[str] | None = None,
) -> tuple[Any, Any]:
    """Build a deep agent with the standard R0 skeleton.

    Parameters
    ----------
    agent_name:
        Used for backend factory name, graph name, and parent_thread_id.
    core_sections:
        Stable prompt sections (identity, instructions, etc.).
    subagents:
        Worker definitions passed to create_deep_agent.
    checkpointer, store:
        LangGraph persistence layer.
    mcp_tools:
        Optional MCP tools to register.
    extra_sections:
        Additional section lists to register (e.g., coding overlays).
    extra_providers:
        Additional context providers beyond the default three.
    configure_tools:
        Optional callback ``(registry: ToolRegistry) -> None`` to register
        additional tools after the standard set.
    conditions:
        Prompt conditions to activate. Defaults to the standard set.
    """
    from deepagents import (
        create_deep_agent as _create,  # pyright: ignore[reportMissingModuleSource]
    )

    llm = build_llm()
    memory_mgr = PersistentMemoryManager(store)

    # --- Tool registry ---
    tool_registry = ToolRegistry()
    register_standard_tools(
        tool_registry,
        memory_mgr,
        store,
        parent_thread_id=f"{agent_name}-global",
        mcp_tools=mcp_tools,
    )
    if configure_tools is not None:
        configure_tools(tool_registry)

    # --- Prompt assembly ---
    section_registry = SectionRegistry()
    section_registry.register_many(core_sections)
    section_registry.register_many(ACTIVATION_SECTIONS)
    for section_list in extra_sections or []:
        section_registry.register_many(section_list)

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

    providers: list[Any] = [
        ThreadContextProvider(),
        MemoryContextProvider(),
        ToolContextProvider(),
        *(extra_providers or []),
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
    active_conditions = conditions or {
        "memory",
        "orchestration",
        "deferred_tools",
        "skills",
        "async_tasks",
    }
    system_prompt = composer.compose_sections_only(conditions=active_conditions)

    # --- Build the deep agent ---
    # ``subagents`` is a list of dict-shaped specs (deepagents accepts both
    # the typed ``SubAgent`` dataclass and the dict form at runtime, but
    # the stub only types the dataclass path).
    graph = _create(
        model=llm,
        tools=tool_registry.compile_tools(),
        system_prompt=system_prompt,
        middleware=middleware,
        subagents=subagents,  # pyright: ignore[reportArgumentType]
        checkpointer=checkpointer,
        store=store,
        backend=build_backend_factory(agent_name),
        name=agent_name,
    )
    return graph, command_dispatcher
