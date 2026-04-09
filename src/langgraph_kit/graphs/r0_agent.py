"""R0 deep agent demonstrating all R0 features.

Integrates: prompt assembly, persistent memory, session notebook,
tool registry, auto memory extraction, context pressure management,
continuation policy, stop hooks, and multi-agent orchestration.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware

from langgraph_kit.core.context_management.pressure import PressureMonitor
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
from langgraph_kit.graphs._builder import (
    build_backend_factory,
    build_command_dispatcher,
    build_middleware_stack,
    register_standard_tools,
)
from langgraph_kit.llm import build_llm

logger = logging.getLogger(__name__)


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
# R0-009: Declarative worker definitions
# ---------------------------------------------------------------------------

WORKER_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "researcher",
        "description": (
            "Deep codebase research and investigation. Use when you need to "
            "explore multiple files, trace execution paths, or understand "
            "architecture across a codebase."
        ),
        "system_prompt": (
            "You are a research specialist. Investigate thoroughly, report "
            "findings in a structured format with file paths and line numbers. "
            "Stay within the assigned scope. Do not make changes — only report."
        ),
    },
    {
        "name": "implementer",
        "description": (
            "Focused code implementation within a bounded scope. Use when the "
            "change is well-understood and the scope is clear."
        ),
        "system_prompt": (
            "You are an implementation specialist. Make the requested changes "
            "precisely and completely. Follow existing code conventions. "
            "Report what you changed and any issues encountered."
        ),
    },
    {
        "name": "verifier",
        "description": (
            "Independent verification of changes. Use after implementation to "
            "check correctness with a fresh perspective."
        ),
        "system_prompt": (
            "You are a verification specialist. Review the changes for "
            "correctness, edge cases, and adherence to requirements. "
            "Do not fix issues — report them clearly so the supervisor "
            "can decide next steps."
        ),
    },
]


# ---------------------------------------------------------------------------
# R0-014: Stop hooks lifecycle
# ---------------------------------------------------------------------------


class StopHooksMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Runs registered lifecycle hooks at turn boundaries.

    Hooks are executed in order after the agent completes.
    Non-blocking hooks log failures; blocking hooks propagate exceptions.
    """

    def __init__(self, hooks: list[Any] | None = None) -> None:
        super().__init__()
        self._hooks: list[Any] = hooks or []

    def register_hook(self, hook: Any) -> None:
        self._hooks.append(hook)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        for hook in self._hooks:
            try:
                if hasattr(hook, "on_turn_complete"):
                    await hook.on_turn_complete(state)
            except Exception:
                blocking = getattr(hook, "blocking", False)
                if blocking:
                    raise
                logger.exception("Non-blocking hook failed: %s", hook)
        return None


# ---------------------------------------------------------------------------
# R0-001: Runtime state tracking middleware
# ---------------------------------------------------------------------------


class RuntimeStateMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Tracks structured state transitions and emits runtime metadata.

    Uses contextvars for per-request state so concurrent invocations don't interfere.
    """

    _cv_state: contextvars.ContextVar[str] = contextvars.ContextVar(
        "runtime_state", default="idle"
    )
    _cv_stop_reason: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "runtime_stop_reason", default=None
    )
    _cv_turn_count: contextvars.ContextVar[int] = contextvars.ContextVar(
        "runtime_turn_count", default=0
    )

    @property
    def state(self) -> str:
        return self._cv_state.get()

    @property
    def stop_reason(self) -> str | None:
        return self._cv_stop_reason.get()

    @property
    def turn_count(self) -> int:
        return self._cv_turn_count.get()

    async def abefore_agent(self, _state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        self._cv_state.set("started")
        self._cv_stop_reason.set(None)
        self._cv_turn_count.set(self._cv_turn_count.get() + 1)
        return None

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self._cv_state.set("streaming")
        try:
            result = await handler(request)
            self._cv_state.set("completed")
            self._cv_stop_reason.set("final_answer")
            return result
        except Exception as exc:
            self._cv_state.set("failed")
            self._cv_stop_reason.set(f"error: {type(exc).__name__}")
            raise

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        prev = self._cv_state.get()
        self._cv_state.set("tool_running")
        try:
            result = await handler(request)
            self._cv_state.set(prev)
            return result
        except Exception:
            self._cv_state.set("tool_failed")
            raise


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_r0_agent(
    checkpointer: Any, store: Any, *, mcp_tools: list[Any] | None = None
) -> Any:
    """Build the R0 demo agent with all features wired together."""
    from deepagents import (
        create_deep_agent,  # pyright: ignore[reportMissingModuleSource]
    )

    llm = build_llm()
    memory_mgr = PersistentMemoryManager(store)

    # --- Tool registry ---
    tool_registry = ToolRegistry()
    register_standard_tools(
        tool_registry,
        memory_mgr,
        store,
        parent_thread_id="r0-global",
        mcp_tools=mcp_tools,
    )

    # --- Prompt assembly ---
    section_registry = SectionRegistry()
    section_registry.register_many(_CORE_SECTIONS)
    section_registry.register_many(ACTIVATION_SECTIONS)

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
        subagents=WORKER_DEFINITIONS,
        checkpointer=checkpointer,
        store=store,
        backend=build_backend_factory("r0_agent"),
        name="r0-agent",
    )
    return graph, command_dispatcher
