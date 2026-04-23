"""Shared agent builder — extracts the common skeleton for deep agent construction.

Both reference_deep_agent and coding_agent follow the same build sequence
(LLM → tools → prompt assembly → middleware → create_deep_agent). This module
captures that skeleton so each agent only specifies its unique overlays.

.. _recursion-limit:

Default recursion limit
-----------------------
Deep agents built by this module default to ``recursion_limit = 100`` (vs.
LangGraph's own default of 25). This is set high because full-stack deep
agents routinely burn through the default on a single real task — prompt
assembly, middleware passes, worker round-trips, and tool loops all count
against the limit, and Pregel raises ``GraphRecursionError`` the moment
it is hit.

**To override**: pass ``recursion_limit=<n>`` to
:func:`build_deep_agent` (or the higher-level wrappers like
``build_reference_deep_agent`` / ``build_coding_agent``) at build time,
or pass ``config={"recursion_limit": <n>}`` at invoke/stream time — the
runtime config wins over the build-time default.

Raise it higher (e.g. 200, 500) for long-running autonomous runs; lower
it to cap runaway loops in tests or evals. See :data:`DEFAULT_RECURSION_LIMIT`.
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

#: Default recursion limit applied to every deep agent built by this module.
#:
#: LangGraph's own default is 25, which is not enough for a full-stack deep
#: agent: a single real task can easily spend that many supersteps on prompt
#: assembly, middleware, worker round-trips, and tool loops, and Pregel raises
#: ``GraphRecursionError`` the moment the limit is hit.
#:
#: **Override per build**:
#: ``build_reference_deep_agent(..., recursion_limit=500)``.
#:
#: **Override per run**:
#: ``graph.ainvoke(input, config={"recursion_limit": 500})``
#: (runtime config wins over the build-time default).
DEFAULT_RECURSION_LIMIT: int = 100


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
    configure_deferred_tools: Any | None = None,
    stop_hooks: list[Any] | None = None,
    conditions: set[str] | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> tuple[Any, Any]:
    """Build a deep agent with the standard skeleton.

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
    configure_deferred_tools:
        Optional callback ``(deferred: DeferredToolRegistry) -> None`` to
        register tools the agent can discover via ``tool_search`` and
        invoke via ``call_deferred_tool``. Deferred tools don't take up
        room in the active tool-binding surface — use this for large or
        rarely-used catalogs.
    stop_hooks:
        Optional list of objects with an ``async on_turn_complete(state)``
        method that run after every agent turn (via
        :class:`StopHooksMiddleware`). Hooks with ``blocking=True``
        propagate exceptions; others are logged and swallowed. Use for
        observability, logging, or cross-turn bookkeeping.
    conditions:
        Prompt conditions to activate. Defaults to the standard set.
    recursion_limit:
        Default LangGraph ``recursion_limit`` bound to the compiled graph
        via ``with_config``. Defaults to
        :data:`DEFAULT_RECURSION_LIMIT` (100). Pass a higher value for
        long-running autonomous runs, or override per-invocation via
        ``graph.ainvoke(..., config={"recursion_limit": N})``. See the
        module docstring for more context.
    """
    from deepagents import (
        create_deep_agent as _create,  # pyright: ignore[reportMissingModuleSource]
    )

    llm = build_llm()
    memory_mgr = PersistentMemoryManager(store)

    # --- Tool registry ---
    tool_registry = ToolRegistry()
    deferred_registry = register_standard_tools(
        tool_registry,
        memory_mgr,
        store,
        parent_thread_id=f"{agent_name}-global",
        mcp_tools=mcp_tools,
    )
    if configure_tools is not None:
        configure_tools(tool_registry)
    if configure_deferred_tools is not None:
        configure_deferred_tools(deferred_registry)

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
        stop_hooks=stop_hooks,
    )

    # --- Compose system prompt ---
    # Auto-activate the "extensions" condition when the caller supplied
    # anything plugin-shaped (MCP tools or extra prompt sections). The
    # ``extension_awareness`` activation section tells the model that
    # plugin-provided capabilities are first-class; gating it on any
    # actual extension avoids bloating the prompt on vanilla builds.
    # Callers passing an explicit ``conditions=`` set stay in control.
    auto_conditions: set[str] = {
        "memory",
        "orchestration",
        "deferred_tools",
        "skills",
        "async_tasks",
    }
    if mcp_tools or extra_sections:
        auto_conditions.add("extensions")
    active_conditions = conditions or auto_conditions
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
    # Bind the default recursion_limit (see DEFAULT_RECURSION_LIMIT). This
    # returns a new CompiledStateGraph with the config merged — Pregel-specific
    # methods like ``aget_state`` are preserved. Runtime config passed to
    # ``invoke``/``ainvoke``/``astream_events`` overrides this default.
    graph = graph.with_config({"recursion_limit": recursion_limit})
    return graph, command_dispatcher
