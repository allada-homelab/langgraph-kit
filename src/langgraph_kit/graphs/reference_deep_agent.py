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
from langgraph_kit.core.resilience.stop_hooks import TurnTelemetryStopHook
from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT, build_deep_agent
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

    return build_deep_agent(
        agent_name="reference-deep-agent",
        core_sections=_CORE_SECTIONS,
        subagents=GENERAL_WORKERS,
        checkpointer=checkpointer,
        store=store,
        mcp_tools=mcp_tools,
        plugins=plugins,
        recursion_limit=recursion_limit,
        stop_hooks=stop_hooks or None,
        configure_deferred_tools=configure_deferred_tools,
    )
