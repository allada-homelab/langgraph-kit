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

import logging
from typing import Any

from langgraph_kit._config import get_config
from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.graph_builder.backend import build_backend_factory
from langgraph_kit.core.graph_builder.commands import build_command_dispatcher
from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
from langgraph_kit.core.graph_builder.tools import register_standard_tools
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.plugins.registry import (
    PluginContribution,
    PluginRegistry,
)
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

logger = logging.getLogger(__name__)

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


def bind_kit_defaults(graph: Any, *, recursion_limit: int) -> Any:
    """Bind kit defaults to a compiled graph such that they survive ``astream_events``.

    ``CompiledStateGraph.with_config({"recursion_limit": N})`` is honored by
    ``ainvoke`` and ``astream`` — those paths run their own
    ``ensure_config(self.config, config)`` inside ``Pregel.astream``. It is
    **not** honored by ``astream_events``: that call goes through
    ``Runnable.astream_events`` →
    ``langchain_core.tracers.event_stream._astream_events_implementation_v2``,
    which calls ``langchain_core.runnables.config.ensure_config(config)`` and
    materializes a default ``recursion_limit=25`` into the dict before
    dispatching to ``Pregel.astream``. Pregel's own merge then treats the 25
    as an explicit caller value and clobbers the bound default.

    We work around it by pre-merging ``self.config`` into the caller's config
    on ``astream_events``, so the bound ``recursion_limit`` is already present
    and non-default by the time ``langchain_core.ensure_config`` fills in
    defaults. Caller-supplied ``config`` entries still win — the langgraph
    variadic ``ensure_config`` applies configs in order with later values
    overriding earlier ones.

    The patch is installed as an instance attribute. ``CompiledStateGraph``
    does not define ``__slots__ = ()``, so this takes precedence over the
    ``Runnable.astream_events`` inherited from the class.
    """
    from langgraph._internal._config import (  # pyright: ignore[reportMissingImports]
        ensure_config as _lg_ensure_config,
    )

    graph = graph.with_config({"recursion_limit": recursion_limit})
    _orig_astream_events = graph.astream_events

    async def _astream_events_with_bound_defaults(
        input_: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        merged = _lg_ensure_config(graph.config, config)
        async for event in _orig_astream_events(input_, config=merged, **kwargs):
            yield event

    graph.astream_events = _astream_events_with_bound_defaults
    return graph


def _coerce_plugin_registry(
    plugins: PluginRegistry | list[PluginContribution] | None,
) -> PluginRegistry | None:
    """Accept the three supported input shapes and return a PluginRegistry.

    - ``None``: no plugins, skip the merge entirely (return ``None``).
    - A pre-built :class:`PluginRegistry`: use it as-is.
    - A list of :class:`PluginContribution`: wrap in a fresh registry so
      the builder only branches on one type downstream.
    """
    if plugins is None:
        return None
    if isinstance(plugins, PluginRegistry):
        return plugins
    registry = PluginRegistry()
    for contrib in plugins:
        registry.register(contrib)
    return registry


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
    tool_search_loop_threshold: int = 5,
    plugins: PluginRegistry | list[PluginContribution] | None = None,
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
    tool_search_loop_threshold:
        Soft loop-detection threshold on consecutive ``tool_search``
        calls. After this many in a row with no other tool call in
        between, the kit appends a non-breaking advisory to the tool's
        return content suggesting the agent try an alternate approach
        (e.g. invoking a previously-discovered tool directly via
        ``call_deferred_tool``). Defaults to 5. Set to ``0`` to disable.
    plugins:
        Plugin contributions to merge into this agent build. Accepts a
        :class:`PluginRegistry` or a bare list of
        :class:`PluginContribution` objects. Each contribution's
        ``tools`` are registered on the active tool surface,
        ``sections`` are added to the prompt assembly, and ``workers``
        are appended to ``subagents``. Plugin tools run BEFORE the
        ``configure_tools`` callback so caller-level overrides win over
        plugin defaults. When any plugin contributes anything the
        ``"extensions"`` prompt condition is auto-activated so the agent
        is told plugin capabilities are first-class.
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
    memory_mgr = PersistentMemoryManager(
        store, embedding_fn=get_config().memory_embedding_fn
    )

    # Normalize the plugin input so the rest of the builder only has to
    # deal with one shape. Accept either a pre-built PluginRegistry or a
    # bare list of PluginContribution (ergonomic for inline cases).
    plugin_registry = _coerce_plugin_registry(plugins)

    # --- Tool registry ---
    tool_registry = ToolRegistry()
    deferred_registry = register_standard_tools(
        tool_registry,
        memory_mgr,
        store,
        parent_thread_id=f"{agent_name}-global",
        mcp_tools=mcp_tools,
    )
    # Plugin tools merge BEFORE the configure_tools callback so the
    # caller's explicit overrides beat plugin defaults on id collisions
    # (ToolRegistry.register is an upsert by capability.id).
    if plugin_registry is not None:
        tool_registry.register_many(plugin_registry.collect_tools())
    if configure_tools is not None:
        configure_tools(tool_registry)
    if configure_deferred_tools is not None:
        configure_deferred_tools(deferred_registry)

    # ``register_standard_tools`` eagerly binds ``tool_search`` +
    # ``call_deferred_tool`` so direct callers (e.g. the CLI-scaffolded
    # builder) get them without extra wiring. Inside this builder we
    # know whether the deferred catalog actually ended up populated — if
    # it didn't, strip both tools from the active surface. Leaving them
    # bound against an empty registry invites suggestible models (Qwen
    # is the reliable reproducer) to spin in a ``tool_search`` loop
    # hunting for capabilities that don't exist, even though the
    # deferred_tools prompt section is already gated off below.
    if not deferred_registry:
        tool_registry.remove("tool_search")
        tool_registry.remove("call_deferred_tool")

    # --- Prompt assembly ---
    section_registry = SectionRegistry()
    section_registry.register_many(core_sections)
    section_registry.register_many(ACTIVATION_SECTIONS)
    # Plugin-contributed sections go between activation sections and
    # caller-supplied extra_sections so explicit user input wins over
    # plugin defaults on id collisions.
    if plugin_registry is not None:
        section_registry.register_many(plugin_registry.collect_sections())
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
        tool_search_loop_threshold=tool_search_loop_threshold,
    )

    # --- Compose system prompt ---
    # Auto-activate capability-awareness conditions based on what is
    # actually wired up. ``extensions`` is added when any plugin-shaped
    # input is present (MCP tools, extra sections, or a populated
    # PluginRegistry). ``deferred_tools`` is added only when the
    # DeferredToolRegistry is populated — activating its prompt section
    # ("use tool_search to discover additional capabilities… don't
    # assume unavailable — search first") against an empty registry
    # pushes the LLM to call a search that always returns nothing, which
    # on recursion-bound runs manifests as spinning on tool_search.
    # Callers passing an explicit ``conditions=`` stay in control, with
    # one exception: if they request "deferred_tools" but the registry
    # is empty the condition is stripped and a warning is logged,
    # because honoring it would produce the same misdirection.
    auto_conditions: set[str] = {
        "memory",
        "orchestration",
        "skills",
        "async_tasks",
    }
    has_plugins = plugin_registry is not None and bool(plugin_registry.list_plugins())
    if mcp_tools or extra_sections or has_plugins:
        auto_conditions.add("extensions")
    if deferred_registry:
        auto_conditions.add("deferred_tools")

    if conditions is None:
        active_conditions = auto_conditions
    else:
        active_conditions = set(conditions)
        if "deferred_tools" in active_conditions and not deferred_registry:
            logger.warning(
                (
                    "build_deep_agent(agent_name=%r): 'deferred_tools' was"
                    " requested in conditions= but the DeferredToolRegistry"
                    " is empty. Dropping the condition to avoid prompting"
                    " the LLM to call tool_search against an empty catalog."
                    " Either remove 'deferred_tools' from conditions or"
                    " pass configure_deferred_tools= to populate the"
                    " registry."
                ),
                agent_name,
            )
            active_conditions.discard("deferred_tools")
    system_prompt = composer.compose_sections_only(conditions=active_conditions)

    # --- Build the deep agent ---
    # Plugin workers extend the caller-supplied ``subagents`` list.
    # Appended (not prepended) so caller-declared workers keep priority
    # in any deepagents routing that walks the list in order.
    merged_subagents: list[dict[str, Any]] = list(subagents)
    if plugin_registry is not None:
        merged_subagents.extend(plugin_registry.collect_workers())

    # ``subagents`` is a list of dict-shaped specs (deepagents accepts both
    # the typed ``SubAgent`` dataclass and the dict form at runtime, but
    # the stub only types the dataclass path).
    graph = _create(
        model=llm,
        tools=tool_registry.compile_tools(),
        system_prompt=system_prompt,
        middleware=middleware,
        subagents=merged_subagents,  # pyright: ignore[reportArgumentType]
        checkpointer=checkpointer,
        store=store,
        backend=build_backend_factory(agent_name),
        name=agent_name,
    )
    # Bind the default recursion_limit (see DEFAULT_RECURSION_LIMIT). Uses
    # ``bind_kit_defaults`` rather than a bare ``with_config`` so the bound
    # default survives the ``astream_events`` codepath — see that helper's
    # docstring for the langchain_core/langgraph config-merge interaction.
    graph = bind_kit_defaults(graph, recursion_limit=recursion_limit)
    return graph, command_dispatcher
