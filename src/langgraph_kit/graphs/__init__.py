"""Agent graph registration.

Called during app startup to compile and register all agent graphs.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.registry import register

logger = logging.getLogger(__name__)


def register_all(
    checkpointer: Any,
    store: Any,
    *,
    mcp_tools: list[Any] | None = None,
) -> None:
    """Discover and register all agent graphs."""
    from langgraph_kit.graphs.echo_agent import build_graph

    register("echo-agent", build_graph(checkpointer, store))
    logger.info("Registered agent: echo-agent")

    try:
        from langgraph_kit.graphs.deep_agent import build_deep_graph

        register("deep-agent", build_deep_graph(checkpointer, store))
        logger.info("Registered agent: deep-agent")
    except Exception:
        logger.info("deep-agent not available — skipping", exc_info=True)

    try:
        from langgraph_kit.graphs.r0_agent import build_r0_agent

        graph, dispatcher = build_r0_agent(
            checkpointer, store, mcp_tools=mcp_tools or []
        )
        register("r0-agent", graph, command_dispatcher=dispatcher)
        logger.info("Registered agent: r0-agent")
    except Exception:
        logger.info("r0-agent not available — skipping", exc_info=True)

    try:
        from langgraph_kit.graphs.coding_agent import build_coding_agent

        graph, dispatcher = build_coding_agent(
            checkpointer, store, mcp_tools=mcp_tools or []
        )
        register("coding-agent", graph, command_dispatcher=dispatcher)
        logger.info("Registered agent: coding-agent")
    except Exception:
        logger.info("coding-agent not available — skipping", exc_info=True)
