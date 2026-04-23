"""Coding agent — extends ``reference_deep_agent`` with coding-profile overlays.

Reuses the reference deep agent's core sections and the shared
``build_deep_agent`` skeleton, and layers on coding-specific prompt
sections, a git context provider, worktree tools, an enhanced verification
worker, and slash-command dispatch. Use this module as the template for
any new domain specialization.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.graph_builder.tools import register_tool
from langgraph_kit.core.orchestration.workers import CODING_WORKERS
from langgraph_kit.core.prompt_assembly.coding_sections import (
    CODING_SEARCH_SECTIONS,
    CODING_WORKFLOW_SECTIONS,
)
from langgraph_kit.core.prompt_assembly.git_context import GitContextProvider
from langgraph_kit.core.tools.capability import ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.core.tools.worktree import (
    WORKTREE_GUIDANCE_SECTION,
    build_worktree_tools,
)
from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT, build_deep_agent
from langgraph_kit.graphs.reference_deep_agent import _CORE_SECTIONS


def _register_worktree_tools(registry: ToolRegistry) -> None:
    """Register coding-specific worktree tools on the registry."""
    for i, tool_fn in enumerate(build_worktree_tools()):
        name = getattr(tool_fn, "__name__", f"worktree_tool_{i}")
        register_tool(
            registry,
            tool_fn,
            id_prefix="worktree",
            tags=["git", "worktree"],
            risk=ToolRisk.READ_ONLY if name == "list_worktrees" else ToolRisk.MUTATING,
        )


def build_coding_agent(
    checkpointer: Any,
    store: Any,
    *,
    mcp_tools: list[Any] | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> Any:
    """Build the coding agent by layering coding-profile overlays onto the reference skeleton.

    ``recursion_limit`` defaults to :data:`DEFAULT_RECURSION_LIMIT` (100);
    pass a higher value for long autonomous runs, or override at invoke
    time with ``config={"recursion_limit": N}``.
    """
    return build_deep_agent(
        agent_name="coding-agent",
        core_sections=_CORE_SECTIONS,
        subagents=CODING_WORKERS,
        checkpointer=checkpointer,
        store=store,
        mcp_tools=mcp_tools,
        extra_sections=[
            CODING_WORKFLOW_SECTIONS,
            CODING_SEARCH_SECTIONS,
            [WORKTREE_GUIDANCE_SECTION],
        ],
        extra_providers=[GitContextProvider()],
        configure_tools=_register_worktree_tools,
        recursion_limit=recursion_limit,
    )
