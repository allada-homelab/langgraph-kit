"""Plugin registry — extensions contribute tools, prompt sections, and workers."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_kit.core.prompt_assembly.sections import PromptSection
from langgraph_kit.core.tools.capability import ToolCapability

logger = logging.getLogger(__name__)


class PluginContribution:
    """What a plugin contributes to the agent system."""

    def __init__(
        self,
        plugin_id: str,
        *,
        tools: list[ToolCapability] | None = None,
        sections: list[PromptSection] | None = None,
        workers: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.plugin_id = plugin_id
        self.tools = tools or []
        self.sections = sections or []
        self.workers = workers or []


class PluginRegistry:
    """Central registry for plugin-provided extensions.

    Plugins register their contributions (tools, prompt sections, workers).
    The agent builder queries the registry to merge contributions into
    the active configuration.
    """

    def __init__(self) -> None:
        super().__init__()
        self._plugins: dict[str, PluginContribution] = {}

    def register(self, contribution: PluginContribution) -> None:
        """Register a plugin's contributions."""
        self._plugins[contribution.plugin_id] = contribution
        logger.info("Plugin registered: %s", contribution.plugin_id)

    def unregister(self, plugin_id: str) -> None:
        self._plugins.pop(plugin_id, None)

    def get(self, plugin_id: str) -> PluginContribution | None:
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> list[str]:
        return list(self._plugins.keys())

    def collect_tools(self) -> list[ToolCapability]:
        """Collect all tool contributions from all plugins."""
        tools: list[ToolCapability] = []
        for contrib in self._plugins.values():
            tools.extend(contrib.tools)
        return tools

    def collect_sections(self) -> list[PromptSection]:
        """Collect all prompt section contributions from all plugins."""
        sections: list[PromptSection] = []
        for contrib in self._plugins.values():
            sections.extend(contrib.sections)
        return sections

    def collect_workers(self) -> list[dict[str, Any]]:
        """Collect all worker definitions from all plugins."""
        workers: list[dict[str, Any]] = []
        for contrib in self._plugins.values():
            workers.extend(contrib.workers)
        return workers
