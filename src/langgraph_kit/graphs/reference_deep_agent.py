"""Reference full-stack deep agent.

Wires every kit feature together: prompt assembly, persistent memory,
session notebook, tool registry, auto memory extraction, context pressure
management (including full compaction), continuation policy, stop hooks,
and multi-agent orchestration. Clone this module when starting a new
domain-specific agent and layer on domain-specific sections, providers,
and tools (see ``coding_agent`` for the canonical extension pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider
from langgraph_kit.core.resilience.stop_hooks import TurnTelemetryStopHook
from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT, build_deep_agent
from langgraph_kit.graphs._reference_custom_tools import (
    make_reference_configure_tools,
)
from langgraph_kit.graphs._reference_deferred_tools import (
    make_reference_deferred_configurator,
)

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
    checkpointer: Any,
    store: Any,
    *,
    mcp_tools: list[Any] | None = None,
    plugins: Any = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    enable_default_stop_hooks: bool = True,
    extra_stop_hooks: list[Any] | None = None,
    enable_default_deferred_tools: bool = True,
    extra_deferred_tools: Any | None = None,
    enable_default_custom_tools: bool = True,
    extra_configure_tools: Any | None = None,
    enable_default_hitl_demo: bool = True,
    enable_default_extra_providers: bool = True,
    extra_providers: list[Any] | None = None,
    output_schema: type[BaseModel] | None = None,
    coordinator: bool = False,
    llm_callbacks: list[Any] | None = None,
) -> Any:
    """Build the reference deep agent with all kit features wired together.

    ``recursion_limit`` defaults to :data:`DEFAULT_RECURSION_LIMIT` (100);
    pass a higher value for long autonomous runs, or override at invoke
    time with ``config={"recursion_limit": N}``.

    ``plugins`` accepts a ``PluginRegistry`` or a list of
    ``PluginContribution`` objects whose tools, prompt sections, and
    worker definitions are merged into the build. See
    :func:`langgraph_kit.graphs._builder.build_deep_agent` for details.

    ``enable_default_stop_hooks`` (default ``True``) registers a
    :class:`~langgraph_kit.core.resilience.stop_hooks.TurnTelemetryStopHook`
    against :class:`StopHooksMiddleware` so the lifecycle-hook path is
    exercised by the reference build. Set to ``False`` to opt out.

    ``extra_stop_hooks`` is appended after the default so callers can
    stack additional hooks alongside (or instead of) the telemetry hook.
    Each entry should expose an awaitable ``on_turn_complete(state)``;
    set ``blocking=True`` on a hook to make its exceptions fail the turn
    rather than be logged and swallowed.

    ``enable_default_deferred_tools`` (default ``True``) populates the
    :class:`~langgraph_kit.core.tools.deferred.DeferredToolRegistry` with
    a small set of demo tools so the ``tool_search`` /
    ``call_deferred_tool`` discovery loop is exercised by default —
    closing the showcase gap where the reference always tripped the
    "empty deferred → strip the search tools" branch in the builder.
    Set to ``False`` to leave the catalog empty.

    ``extra_deferred_tools`` is a callback
    ``(DeferredToolRegistry) -> None`` that runs *after* the default
    registration. Because the registry is keyed by capability id, a
    callback registering under a default id (e.g. ``"ref_web_fetch_demo"``)
    overrides the demo — keeping the "caller wins on collisions"
    precedence the rest of the builder follows.

    ``enable_default_custom_tools`` (default ``True``) registers a
    minimal ``current_environment`` tool via the ``configure_tools=``
    extension point. The tool is read-only and side-effect-free; its
    purpose is to demonstrate how domain agents can layer custom tools
    onto the reference build (see ``coding_agent`` for a
    production-shaped example with worktree tools).

    ``extra_configure_tools`` is a callback
    ``(ToolRegistry) -> None`` that runs *after* the default tool
    registration so caller registrations under matching ids override
    the defaults — same upsert-by-id precedence as plugin tools.

    ``enable_default_hitl_demo`` (default ``True``) registers a
    ``confirm_destructive_demo`` capability whose
    ``interrupt_before=True`` flag triggers
    :class:`~langgraph_kit.core.hitl.auto_interrupt.AutoInterruptMiddleware`
    on every call — exercising the HITL gating path that otherwise
    has no shipped tool to drive it. Toggleable independently from
    ``enable_default_custom_tools`` so callers who want the
    diagnostic tool but no HITL flow can opt out.

    ``enable_default_extra_providers`` (default ``True``) registers a
    :class:`~langgraph_kit.core.prompt_assembly.system_context.SystemContextProvider`
    on the prompt composer, mirroring how ``coding_agent`` ships
    ``GitContextProvider``. The provider injects current UTC datetime,
    platform name, OS name, and the kit version under a
    ``# System Context`` heading.

    ``extra_providers`` is appended after the default so callers can
    stack additional :class:`~langgraph_kit.core.prompt_assembly.context_providers.ContextProvider`
    instances. Each provider's ``async provide(context)`` is invoked
    by :class:`PromptComposer` when a full prompt is composed.

    ``output_schema`` accepts a Pydantic ``BaseModel`` subclass. When
    set, :class:`~langgraph_kit.core.resilience.structured_output.StructuredOutputMiddleware`
    is appended to the middleware stack and validates the agent's
    terminal message against the schema (looking for a single
    ``<output_schema>{...}</output_schema>`` block). On mismatch the
    middleware injects a retry nudge with the JSON-Schema rendering
    of the model. ``None`` (default) leaves structured-output
    validation off.

    ``coordinator`` (default ``False``) flips the build into
    coordinator mode (see :class:`~langgraph_kit.core.coordinator.CoordinatorMode`):
    the active tool surface is narrowed to ``ToolRisk.READ_ONLY``
    capabilities + the delegation tools, and coordinator-specific
    prompt sections are merged. Use the
    :func:`build_reference_coordinator_agent` convenience wrapper
    when you want this profile by default.

    ``llm_callbacks`` are bound to the LLM via ``with_config`` so they
    participate in every model call the graph makes. Use this entry
    point for cost / budget instrumentation
    (:class:`~langgraph_kit.core.cost.callback.TokenTrackingCallback`),
    Langfuse handlers, or any other LangChain
    :class:`~langchain_core.callbacks.AsyncCallbackHandler`. The
    caller owns the callback object so they can poll its counters
    after the run; pair with
    :class:`~langgraph_kit.core.cost.budget.BudgetManager` for
    per-thread budget enforcement on the Store.
    """
    stop_hooks: list[Any] = []
    if enable_default_stop_hooks:
        stop_hooks.append(TurnTelemetryStopHook())
    if extra_stop_hooks:
        stop_hooks.extend(extra_stop_hooks)

    if enable_default_deferred_tools:
        configure_deferred_tools = make_reference_deferred_configurator(
            extra=extra_deferred_tools
        )
    elif extra_deferred_tools is not None:
        configure_deferred_tools = extra_deferred_tools
    else:
        configure_deferred_tools = None

    if enable_default_custom_tools:
        configure_tools = make_reference_configure_tools(
            extra=extra_configure_tools,
            include_hitl_demo=enable_default_hitl_demo,
        )
    elif extra_configure_tools is not None:
        configure_tools = extra_configure_tools
    else:
        configure_tools = None

    providers: list[Any] = []
    if enable_default_extra_providers:
        providers.append(SystemContextProvider())
    if extra_providers:
        providers.extend(extra_providers)

    return build_deep_agent(
        agent_name=("reference-coordinator" if coordinator else "reference-deep-agent"),
        core_sections=_CORE_SECTIONS,
        subagents=GENERAL_WORKERS,
        checkpointer=checkpointer,
        store=store,
        mcp_tools=mcp_tools,
        plugins=plugins,
        recursion_limit=recursion_limit,
        stop_hooks=stop_hooks or None,
        configure_tools=configure_tools,
        configure_deferred_tools=configure_deferred_tools,
        extra_providers=providers or None,
        output_schema=output_schema,
        coordinator=coordinator,
        llm_callbacks=llm_callbacks,
    )


def build_reference_coordinator_agent(
    checkpointer: Any,
    store: Any,
    **kwargs: Any,
) -> Any:
    """Build the reference deep agent in coordinator mode.

    Thin wrapper over :func:`build_reference_deep_agent` with
    ``coordinator=True``. Use this when the agent should delegate
    work via :class:`~langgraph_kit.core.coordinator.CoordinatorMode`
    rather than execute mutating operations itself: the bound tool
    surface is narrowed to ``ToolRisk.READ_ONLY`` capabilities + the
    delegation ``task`` tool, and the system prompt picks up the
    coordinator's delegation / synthesis sections.

    All other kwargs are forwarded as-is to
    :func:`build_reference_deep_agent`, so the same opt-outs
    (``enable_default_stop_hooks``, ``enable_default_deferred_tools``,
    ``enable_default_custom_tools``, ``enable_default_extra_providers``)
    apply.
    """
    if "coordinator" in kwargs:
        msg = (
            "build_reference_coordinator_agent forces coordinator=True; "
            "remove the kwarg or call build_reference_deep_agent directly."
        )
        raise TypeError(msg)
    return build_reference_deep_agent(checkpointer, store, coordinator=True, **kwargs)
