"""Agent graph registration.

Called during app startup to compile and register all agent graphs.

All deep agents built by this package default to
:data:`DEFAULT_RECURSION_LIMIT` (100). Override per-build with
``recursion_limit=<n>`` on any ``build_*_deep_agent`` /
``build_coding_agent`` call, or per-invocation with
``config={"recursion_limit": <n>}``.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.graph_builder.backend import build_backend_factory
from langgraph_kit.core.graph_builder.commands import build_command_dispatcher
from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
from langgraph_kit.core.graph_builder.tools import register_standard_tools
from langgraph_kit.graphs._builder import (
    DEFAULT_RECURSION_LIMIT,
    bind_kit_defaults,
    build_deep_agent,
)
from langgraph_kit.registry import AgentMetadata, register

logger = logging.getLogger(__name__)

# Public re-exports for template-generated agents and out-of-tree consumers.
# The CLI scaffolder imports these symbols; keeping an explicit __all__ here
# gives us a stable surface even if the internal layout moves.
__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "bind_kit_defaults",
    "build_backend_factory",
    "build_command_dispatcher",
    "build_deep_agent",
    "build_middleware_stack",
    "register_all",
    "register_standard_tools",
]


def register_all(
    checkpointer: Any,
    store: Any,
    *,
    mcp_tools: list[Any] | None = None,
) -> None:
    """Discover and register all agent graphs."""
    from langgraph_kit.graphs.echo_agent import build_graph

    register(
        "echo-agent",
        build_graph(checkpointer, store),
        metadata=AgentMetadata(
            description="Minimal echo agent for testing and demonstration",
            tags=["testing", "echo"],
            capabilities=["streaming"],
        ),
    )
    logger.info("Registered agent: echo-agent")

    try:
        from langgraph_kit.graphs.basic_deep_agent import build_basic_deep_agent

        register(
            "basic-deep-agent",
            build_basic_deep_agent(checkpointer, store),
            metadata=AgentMetadata(
                description=(
                    "Minimal deepagents example — framework defaults with a "
                    "generic system prompt. See reference-deep-agent for the "
                    "full-featured version."
                ),
                tags=["reasoning", "deep", "example"],
                capabilities=["streaming", "multi-agent"],
            ),
        )
        logger.info("Registered agent: basic-deep-agent")
    except Exception:
        logger.info("basic-deep-agent not available — skipping", exc_info=True)

    try:
        from langgraph_kit.graphs.reference_deep_agent import (
            build_reference_deep_agent,
        )

        graph, dispatcher = build_reference_deep_agent(
            checkpointer, store, mcp_tools=mcp_tools or []
        )
        register(
            "reference-deep-agent",
            graph,
            command_dispatcher=dispatcher,
            metadata=AgentMetadata(
                description=(
                    "Reference full-stack deep agent wiring every kit feature "
                    "(prompt assembly, persistent memory, tool registry, "
                    "middleware, workers, slash commands). Clone this when "
                    "starting a new domain agent."
                ),
                tags=["general", "memory", "tools", "orchestration", "reference"],
                capabilities=["streaming", "hitl", "memory", "mcp", "commands"],
            ),
        )
        logger.info("Registered agent: reference-deep-agent")
    except Exception:
        logger.info("reference-deep-agent not available — skipping", exc_info=True)

    try:
        from langgraph_kit.graphs.coding_agent import build_coding_agent

        graph, dispatcher = build_coding_agent(
            checkpointer, store, mcp_tools=mcp_tools or []
        )
        register(
            "coding-agent",
            graph,
            command_dispatcher=dispatcher,
            metadata=AgentMetadata(
                description="Specialized coding agent with git context, worktree tools, and code review",
                tags=["coding", "git", "tools"],
                capabilities=["streaming", "hitl", "memory", "mcp", "commands"],
            ),
        )
        logger.info("Registered agent: coding-agent")
    except Exception:
        logger.info("coding-agent not available — skipping", exc_info=True)

    try:
        from langgraph_kit.graphs.supervisor_agent import (
            SUPERVISOR_AGENT_ID,
            build_supervisor_agent,
        )

        graph = build_supervisor_agent(checkpointer, store)
        register(
            SUPERVISOR_AGENT_ID,
            graph,
            metadata=AgentMetadata(
                description="Routes requests to the best available specialist agent",
                tags=["routing", "orchestration", "multi-agent"],
                capabilities=["streaming", "delegation"],
            ),
        )
        logger.info("Registered agent: %s", SUPERVISOR_AGENT_ID)
    except Exception:
        logger.info("supervisor-agent not available — skipping", exc_info=True)
