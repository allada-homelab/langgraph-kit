"""Sample plugin used by ``examples/plugins_skill_discovery.py``.

A real plugin is a ``.py`` file (or package) with a top-level
``contribute(**kwargs) -> PluginContribution`` function. Drop it into
``AgentConfig.plugins_dir`` and the loader picks it up at startup.

This file is intentionally named with a leading underscore so the
:mod:`examples.run_all` smoke driver doesn't try to execute it as a
standalone demo.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.plugins.registry import PluginContribution
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk


def _stub_lookup(query: str) -> str:
    """Read-only stub — returns a deterministic answer for the demo."""
    return f"[stub] would have searched for: {query}"


def contribute(**_kwargs: Any) -> PluginContribution:
    """Plugin entry point. The loader calls this at registration time."""
    return PluginContribution(
        plugin_id="sample-plugin",
        tools=[
            ToolCapability(
                id="sample_lookup",
                name="sample_lookup",
                description="Look up a value (sample plugin demo).",
                fn=_stub_lookup,
                risk=ToolRisk.READ_ONLY,
                tags=["sample"],
                prompt_guidance="Use to demonstrate plugin loading.",
            )
        ],
    )
