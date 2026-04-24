"""Regression tests for Phase N skills / plugins fixes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest  # noqa: TC002 — used at runtime via pytest.LogCaptureFixture type

from langgraph_kit.core.plugins.loader import PluginLoader
from langgraph_kit.core.skills.models import SkillMetadata
from langgraph_kit.core.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill(dir_: Path, name: str, frontmatter: str) -> None:
    skill_dir = dir_ / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\nBody for {name}.",
        encoding="utf-8",
    )


def test_skill_registry_get_is_case_insensitive(tmp_path: Path) -> None:
    _write_skill(tmp_path, "research", "name: research\ndescription: r")
    registry = SkillRegistry()
    registry.load_from_directory(tmp_path)

    # Both the original-case and mixed-case queries resolve the same record.
    assert registry.get("research") is not None
    assert registry.get("Research") is not None
    assert registry.get("RESEARCH") is not None


def test_skill_registry_isolates_one_bad_skill(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One malformed skill used to abort the whole scan — it should
    now be skipped with a warning while the valid skills still load."""
    # Valid skill.
    _write_skill(tmp_path, "good", "name: good\ndescription: ok")
    # Malformed: tags must be a list, not a scalar string. Previously
    # this raised ValidationError and crashed load_from_directory.
    _write_skill(
        tmp_path,
        "bad",
        "name: bad\ndescription: d\ntags: not-a-list",
    )

    registry = SkillRegistry()
    with caplog.at_level(logging.WARNING):
        count = registry.load_from_directory(tmp_path)

    assert count == 1
    assert registry.get("good") is not None
    assert registry.get("bad") is None


def test_allowed_tools_field_is_removed() -> None:
    """The field was defined but never enforced, so it misled callers.
    The regression guards against anyone re-adding it without wiring."""
    assert "allowed_tools" not in SkillMetadata.model_fields


def test_plugin_loader_logs_concrete_error_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plugin load failures must identify the exception type in the
    WARNING line so operators can debug without scraping the full
    traceback log."""
    plugin = tmp_path / "broken.py"
    plugin.write_text(
        "def contribute(**kwargs):\n    raise ImportError('missing widget')\n",
        encoding="utf-8",
    )

    loader = PluginLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_from_directory(tmp_path)

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "ImportError" in messages
    assert "missing widget" in messages
