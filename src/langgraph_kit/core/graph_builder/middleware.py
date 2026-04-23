"""Middleware stack assembly for agent graph builders."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.commands.dispatch import CommandDispatcher
from langgraph_kit.core.commands.middleware import CommandMiddleware
from langgraph_kit.core.context_management.pressure import PressureMonitor
from langgraph_kit.core.context_management.pressure_middleware import (
    PressureMiddleware,
)
from langgraph_kit.core.context_management.result_persistence import (
    ResultPersistenceMiddleware,
)
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
    ToolErrorMiddleware,
    ToolLoopGuardMiddleware,
)


def build_middleware_stack(
    *,
    llm: Any,
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    command_dispatcher: CommandDispatcher | None = None,
    stop_hooks: list[Any] | None = None,
    tool_search_loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
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
    """
    middleware: list[Any] = []

    # Command interception (if dispatcher provided)
    if command_dispatcher:
        middleware.append(CommandMiddleware(command_dispatcher))

    middleware.extend(
        [
            RuntimeStateMiddleware(),
            QueuedInputMiddleware(),
            ToolErrorMiddleware(max_retries=1),
            # Loop guard wraps tool calls BEFORE ResultPersistence so the
            # advisory appears in the full-text content rather than in
            # the persisted preview, and the model sees it on its next
            # read of the tool result.
            ToolLoopGuardMiddleware(
                tool_name="tool_search",
                threshold=tool_search_loop_threshold,
            ),
            PressureMiddleware(pressure_monitor, llm=llm),
            ResultPersistenceMiddleware(),
            ExtractionMiddleware(
                AutoMemoryExtractor(memory_mgr, llm), scope=MemoryScope.USER
            ),
            EmptyTurnMiddleware(max_nudges=2),
            CompletionGuardMiddleware(min_tool_calls=1),
            StopHooksMiddleware(hooks=stop_hooks),
            PostRunBackstopMiddleware(),
        ]
    )

    return middleware, pressure_monitor
