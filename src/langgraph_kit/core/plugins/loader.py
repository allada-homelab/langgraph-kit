"""Plugin loader — discovers and loads plugins from a directory.

Each plugin is a Python module (``.py`` file) or package (directory with
``__init__.py``) that exports a ``contribute()`` function::

    # plugins/my_plugin.py
    from langgraph_kit.core.plugins.registry import PluginContribution
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    def contribute(**kwargs) -> PluginContribution:
        return PluginContribution(
            plugin_id="my_plugin",
            tools=[ToolCapability(id="my_tool", name="my_tool", ...)],
        )
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies
from typing import Any

from langgraph_kit.core.plugins.registry import PluginContribution, PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Discovers and loads plugins from a directory into a PluginRegistry."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self._registry = registry or PluginRegistry()

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

    def load_from_directory(
        self, path: Path, **kwargs: Any
    ) -> list[PluginContribution]:
        """Load all plugins from a directory.

        Each ``.py`` file (or package) in the directory must export a
        ``contribute(**kwargs) -> PluginContribution`` function.
        kwargs are passed through to each plugin's contribute function
        (e.g. ``store=store, llm=llm``).

        Returns the list of successfully loaded contributions.
        """
        if not path.is_dir():
            logger.debug("Plugin directory does not exist: %s", path)
            return []

        loaded: list[PluginContribution] = []
        for item in sorted(path.iterdir()):
            if item.name.startswith("_"):
                continue
            if item.suffix == ".py" or (
                item.is_dir() and (item / "__init__.py").exists()
            ):
                contribution = self._load_one(item, **kwargs)
                if contribution is not None:
                    self._registry.register(contribution)
                    loaded.append(contribution)

        logger.info("Loaded %d plugin(s) from %s", len(loaded), path)
        return loaded

    def _load_one(self, path: Path, **kwargs: Any) -> PluginContribution | None:
        """Load a single plugin module and call its contribute() function."""
        module_name = f"langgraph_kit_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                path if path.suffix == ".py" else path / "__init__.py",
            )
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin spec: %s", path)
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            contribute_fn = getattr(module, "contribute", None)
            if contribute_fn is None:
                logger.warning(
                    "Plugin '%s' has no contribute() function — skipping", path.name
                )
                return None

            contribution = contribute_fn(**kwargs)
            if not isinstance(contribution, PluginContribution):
                logger.warning(
                    "Plugin '%s' contribute() did not return PluginContribution — skipping",
                    path.name,
                )
                return None

            logger.info("Loaded plugin: %s (%s)", contribution.plugin_id, path.name)
            return contribution

        except Exception:
            logger.exception("Failed to load plugin: %s", path.name)
            return None
