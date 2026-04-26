"""Demo custom tool wired into the reference deep agent.

This module exists to give cloners of ``reference_deep_agent`` an
in-tree, minimal exemplar of the ``configure_tools=`` extension point.
``coding_agent`` ships a production-shaped example (worktree tools);
this is the no-domain version: a single read-only tool that reports
basic process metadata.

The tool is intentionally narrow — clone the registration helper and
swap the body when adding your own.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from langgraph_kit.core.graph_builder.tools import register_tool
from langgraph_kit.core.tools.capability import ToolRisk
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


def register_reference_custom_tools(registry: ToolRegistry) -> None:
    """Register the demo ``current_environment`` tool on ``registry``."""
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


def make_reference_configure_tools(
    extra: Any | None = None,
) -> Any:
    """Build a ``configure_tools=`` callback for the reference build.

    The callback registers the default demo tool first, then runs the
    optional ``extra`` callback. ``ToolRegistry.register`` is an upsert
    on capability id, so anything ``extra`` registers under the demo's
    id (``"reference_current_environment"``) overrides the demo —
    matching the documented "caller wins on collisions" precedence.
    """

    def _configure(registry: ToolRegistry) -> None:
        register_reference_custom_tools(registry)
        if extra is not None:
            extra(registry)

    return _configure
