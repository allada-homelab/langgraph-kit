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
    CompletionGuardMiddleware,
    EmptyTurnMiddleware,
    PostRunBackstopMiddleware,
    RuntimeStateMiddleware,
    StopHooksMiddleware,
    ToolErrorMiddleware,
)


def build_middleware_stack(
    *,
    llm: Any,
    memory_mgr: PersistentMemoryManager,
    pressure_monitor: PressureMonitor,
    command_dispatcher: CommandDispatcher | None = None,
) -> tuple[list[Any], PressureMonitor]:
    """Build the standard middleware stack shared by all deep agents.

    Returns (middleware_list, pressure_monitor) so callers can access
    the monitor for prompt composition.
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
            PressureMiddleware(pressure_monitor),
            ResultPersistenceMiddleware(),
            ExtractionMiddleware(
                AutoMemoryExtractor(memory_mgr, llm), scope=MemoryScope.USER
            ),
            EmptyTurnMiddleware(max_nudges=2),
            CompletionGuardMiddleware(min_tool_calls=1),
            StopHooksMiddleware(),
            PostRunBackstopMiddleware(),
        ]
    )

    return middleware, pressure_monitor
