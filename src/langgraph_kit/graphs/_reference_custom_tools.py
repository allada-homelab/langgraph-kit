"""Demo custom tools wired into the reference deep agent.

This module exists to give cloners of ``reference_deep_agent`` an
in-tree, minimal exemplar of the ``configure_tools=`` extension point.
``coding_agent`` ships a production-shaped example (worktree tools);
this is the no-domain version. Two demos:

- ``current_environment`` — read-only tool reporting Python version,
  OS, kit version, and whether the working directory is a git repo.
  Demonstrates a typical ``configure_tools=`` registration via the
  ``register_tool`` helper.
- ``confirm_destructive_demo`` — a no-op stub representing a
  destructive action whose capability declares
  ``interrupt_before=True``. Demonstrates HITL gating via
  :class:`~langgraph_kit.core.hitl.auto_interrupt.AutoInterruptMiddleware`.

The tools are intentionally narrow — clone the registration helper
and swap the bodies when adding your own.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from langgraph_kit.core.graph_builder.tools import register_tool
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry


async def current_environment() -> str:
    """Report basic info about the agent's runtime: Python version,
    OS, kit version, and whether the working directory is a git repo.

    Use this when the user asks about the agent's environment, or when
    you need to surface kit / Python version info for diagnostics.
    Returns a plain-text summary; do not parse the structure
    programmatically.
    """
    try:
        from langgraph_kit import __version__ as kit_version
    except ImportError:
        kit_version = "unknown"

    git_present = (Path.cwd() / ".git").exists()
    return (
        f"python={platform.python_version()} "
        f"system={platform.system()} "
        f"kit={kit_version} "
        f"git_repo={git_present}"
    )


async def confirm_destructive_demo(target: str) -> str:
    """Simulate a destructive action against ``target`` and return a confirmation string.

    HITL demo: the capability registered for this tool sets
    ``interrupt_before=True``, so ``AutoInterruptMiddleware`` pauses
    the run and surfaces an approval prompt before this body executes.
    The body itself is a no-op — replace it with a real destructive
    action when adapting the pattern.
    """
    return f"[demo] would perform destructive action against: {target}"


def register_reference_custom_tools(registry: ToolRegistry) -> None:
    """Register the read-only ``current_environment`` demo tool on ``registry``."""
    register_tool(
        registry,
        current_environment,
        id_prefix="reference",
        name="current_environment",
        tags=["reference", "environment", "diagnostics"],
        risk=ToolRisk.READ_ONLY,
        prompt_guidance=(
            "Use current_environment to report the agent's Python "
            "version, OS, kit version, and whether the working "
            "directory is a git repo. Read-only and side-effect-free."
        ),
    )


def register_reference_hitl_demo_tool(registry: ToolRegistry) -> None:
    """Register the ``confirm_destructive_demo`` tool with ``interrupt_before=True``.

    The kit's ``register_tool`` helper does not currently forward
    ``interrupt_before`` (its primary use is in the ``__init__`` of
    each tool builder, where the flag is rarely needed). Register
    directly via ``ToolCapability`` so the HITL gating flag flows
    through to ``AutoInterruptMiddleware``.
    """
    registry.register(
        ToolCapability(
            id="reference_confirm_destructive_demo",
            name="confirm_destructive_demo",
            description=(
                "Demo HITL-gated tool. The agent must declare a target; "
                "the user is then prompted to approve before the action "
                "runs. The body is a no-op — replace with a real "
                "destructive action when adapting this pattern."
            ),
            fn=confirm_destructive_demo,
            tags=["reference", "hitl", "demo"],
            risk=ToolRisk.MUTATING,
            interrupt_before=True,
            prompt_guidance=(
                "Use confirm_destructive_demo to demonstrate the HITL "
                "approval flow. The middleware will pause execution and "
                "ask the user to approve before the action runs."
            ),
        )
    )


def make_reference_configure_tools(
    extra: Any | None = None,
    *,
    include_hitl_demo: bool = True,
) -> Any:
    """Build a ``configure_tools=`` callback for the reference build.

    The callback registers the default demo tools (read-only +
    optional HITL-gated), then runs the optional ``extra`` callback.
    ``ToolRegistry.register`` is an upsert on capability id, so
    anything ``extra`` registers under a default id overrides the
    demo — matching the documented "caller wins on collisions"
    precedence.

    ``include_hitl_demo`` toggles the HITL-gated tool independently
    from the read-only demo so callers can keep the read-only tool
    while opting out of the interrupt flow.
    """

    def _configure(registry: ToolRegistry) -> None:
        register_reference_custom_tools(registry)
        if include_hitl_demo:
            register_reference_hitl_demo_tool(registry)
        if extra is not None:
            extra(registry)

    return _configure
