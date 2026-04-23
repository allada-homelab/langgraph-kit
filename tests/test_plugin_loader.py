"""Coverage fill — ``PluginLoader.load_from_directory`` branches.

The loader walks a directory, imports each module that looks like a
plugin, calls its ``contribute(**kwargs) -> PluginContribution``
factory, and registers the result. Tests cover:

- Happy path (single .py plugin + package plugin).
- Plugin with no ``contribute()`` function → logged + skipped.
- Plugin that throws during import → logged + skipped.
- Plugin that returns the wrong type → logged + skipped.
- Missing directory → returns empty.
- Files starting with underscore are ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph_kit.core.plugins.loader import PluginLoader

if TYPE_CHECKING:
    from pathlib import Path


def _write_plugin(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_from_missing_directory_returns_empty(tmp_path: Path) -> None:
    loader = PluginLoader()
    assert loader.load_from_directory(tmp_path / "does-not-exist") == []


def test_load_from_empty_directory_returns_empty(tmp_path: Path) -> None:
    loader = PluginLoader()
    assert loader.load_from_directory(tmp_path) == []


def test_happy_path_py_file_loads_and_registers(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "widget.py",
        """
from langgraph_kit.core.plugins.registry import PluginContribution

def contribute(**kwargs):
    return PluginContribution(plugin_id="widget-plugin")
""",
    )
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].plugin_id == "widget-plugin"
    # Registry side-effect: plugin is registered.
    assert "widget-plugin" in loader.registry.list_plugins()


def test_package_plugin_with_init_py_is_loaded(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    _write_plugin(
        pkg_dir / "__init__.py",
        """
from langgraph_kit.core.plugins.registry import PluginContribution

def contribute(**kwargs):
    return PluginContribution(plugin_id="pkg-plugin")
""",
    )
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    assert [c.plugin_id for c in loaded] == ["pkg-plugin"]


def test_plugin_without_contribute_function_is_skipped(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "silent.py", "# no contribute function here\n")
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    assert loaded == []


def test_plugin_contribute_returning_wrong_type_is_skipped(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "bad_return.py",
        "def contribute(**kwargs):\n    return 'not a contribution'\n",
    )
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    assert loaded == []


def test_plugin_raising_on_import_is_caught_and_logged(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "explodes.py", "raise RuntimeError('boom')\n")
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    assert loaded == []


def test_underscore_prefixed_files_are_skipped(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "_private.py",
        "from langgraph_kit.core.plugins.registry import PluginContribution\n"
        "def contribute(**kwargs):\n"
        "    return PluginContribution(plugin_id='private')\n",
    )
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path)
    # Underscore-prefixed files are dunder/private module convention —
    # the loader skips them.
    assert loaded == []


def test_loader_passes_kwargs_to_contribute(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "reads_kwargs.py",
        """
from langgraph_kit.core.plugins.registry import PluginContribution

def contribute(**kwargs):
    marker = kwargs.get("marker", "missing")
    return PluginContribution(plugin_id=f"kw-{marker}")
""",
    )
    loader = PluginLoader()
    loaded = loader.load_from_directory(tmp_path, marker="passed-through")
    assert [c.plugin_id for c in loaded] == ["kw-passed-through"]


def test_loader_registers_into_caller_supplied_registry(tmp_path: Path) -> None:
    from langgraph_kit.core.plugins.registry import PluginRegistry

    _write_plugin(
        tmp_path / "a.py",
        "from langgraph_kit.core.plugins.registry import PluginContribution\n"
        "def contribute(**kwargs):\n"
        "    return PluginContribution(plugin_id='a')\n",
    )
    outside = PluginRegistry()
    loader = PluginLoader(registry=outside)
    loader.load_from_directory(tmp_path)
    assert "a" in outside.list_plugins()
