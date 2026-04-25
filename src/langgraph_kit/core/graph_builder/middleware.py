"""Middleware stack assembly for agent graph builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_kit.core.commands.dispatch import CommandDispatcher
from langgraph_kit.core.commands.middleware import CommandMiddleware
from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.context_management.pressure_middleware import (
    PressureMiddleware,
)
from langgraph_kit.core.context_management.result_persistence import (
    ResultPersistenceMiddleware,
)
from langgraph_kit.core.hitl import AutoInterruptMiddleware
from langgraph_kit.core.memory.extraction import AutoMemoryExtractor
from langgraph_kit.core.memory.extraction_middleware import ExtractionMiddleware
from langgraph_kit.core.memory.models import MemoryScope
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.orchestration.queue import QueuedInputMiddleware
from langgraph_kit.core.resilience import (
    DEFAULT_LOOP_THRESHOLD,
    CompletionGuardMiddleware,
    EmptyTurnMiddleware,
    PostRunBackstopMiddleware,
    RuntimeStateMiddleware,
    StopHooksMiddleware,
    StructuredOutputMiddleware,
    ToolErrorMiddleware,
    ToolLoopGuardMiddleware,
)
from langgraph_kit.core.security import PromptInjectionGuardMiddleware

if TYPE_CHECKING:
    from pydantic import BaseModel

    from langgraph_kit.core.tools.registry import ToolRegistry


def build_middleware_stack(
    *,
    llm: Any,
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    command_dispatcher: CommandDispatcher | None = None,
    stop_hooks: list[Any] | None = None,
    tool_search_loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
    output_schema: type[BaseModel] | None = None,
    tool_registry: ToolRegistry | None = None,
) -> tuple[list[Any], PressureMonitor]:
    """Build the standard middleware stack shared by all deep agents.

    Returns (middleware_list, pressure_monitor) so callers can access
    the monitor for prompt composition.

    ``stop_hooks`` is forwarded to :class:`StopHooksMiddleware`; hooks
    with an ``on_turn_complete(state)`` coroutine run after every agent
    turn.

    ``tool_search_loop_threshold`` controls the soft loop-detection
    nudge emitted after ``N`` consecutive ``tool_search`` calls.
    Defaults to :data:`DEFAULT_LOOP_THRESHOLD`. Set to ``0`` to disable.

    ``output_schema`` is opt-in: when a Pydantic model class is supplied,
    a :class:`StructuredOutputMiddleware` is appended to validate the
    agent's terminal message and retry on schema mismatch (capped). When
    ``None`` (default), no validation middleware is added — agents that
    return free-form text are unaffected.
    """
    middleware: list[Any] = []

    # Inbound prompt-injection scanner runs early so the flag is
    # available to downstream middleware that wants to react. Mode is
    # read from the active AgentConfig; default ``"warn"`` is
    # non-blocking. ``"off"`` skips appending the middleware entirely
    # so there is zero overhead when the feature is disabled.
    from langgraph_kit._config import get_config

    pi_mode = get_config().prompt_injection_mode
    if pi_mode and pi_mode != "off":
        middleware.append(PromptInjectionGuardMiddleware(mode=pi_mode))  # type: ignore[arg-type]

    # Command interception (if dispatcher provided)
    if command_dispatcher:
        middleware.append(CommandMiddleware(command_dispatcher))

    middleware.extend(
        [
            RuntimeStateMiddleware(),
            QueuedInputMiddleware(),
            ToolErrorMiddleware(max_retries=1),
            # Auto-interrupt sits BEFORE the tool runs, so it has to be
            # outermost on the tool-wrapping side after the error wrap
            # (errors happen during execution; the interrupt prevents
            # execution entirely). Tools without ``interrupt_before``
            # set on their capability pass through unchanged.
            AutoInterruptMiddleware(tool_registry=tool_registry),
            # Loop guard wraps tool calls BEFORE ResultPersistence so the
            # advisory appears in the full-text content rather than in
            # the persisted preview, and the model sees it on its next
            # read of the tool result.
            ToolLoopGuardMiddleware(
                tool_name="tool_search",
                threshold=tool_search_loop_threshold,
            ),
            PressureMiddleware(pressure_monitor, llm=llm),
            ResultPersistenceMiddleware(tool_registry=tool_registry),
            ExtractionMiddleware(
                AutoMemoryExtractor(memory_mgr, llm), scope=MemoryScope.USER
            ),
            EmptyTurnMiddleware(max_nudges=2),
            CompletionGuardMiddleware(min_tool_calls=1),
            StopHooksMiddleware(hooks=stop_hooks),
        ]
    )

    # Structured-output validation slots after the empty-turn / completion
    # guards (they ensure a turn exists at all) and before PostRunBackstop
    # (the schema check is a richer terminal-state check than the backstop's
    # generic last-resort message).
    if output_schema is not None:
        middleware.append(StructuredOutputMiddleware(schema=output_schema))

    middleware.append(PostRunBackstopMiddleware())

    return middleware, pressure_monitor
