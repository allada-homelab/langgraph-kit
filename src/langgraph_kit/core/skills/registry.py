"""Skill registry — discovers and indexes SKILL.md files from directories."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from langgraph_kit.core.skills.models import SkillMetadata

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill_md(path: Path) -> SkillMetadata | None:
    """Parse a SKILL.md file and extract YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Cannot read skill file: %s", path)
        return None

    match = _FRONTMATTER_RE.match(text)
    if not match:
        logger.warning("No YAML frontmatter in %s", path)
        return None

    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        logger.warning("Invalid YAML in %s", path, exc_info=True)
        return None

    if not isinstance(meta, dict) or "name" not in meta:
        logger.warning("Missing 'name' in frontmatter: %s", path)
        return None

    meta_typed: dict[str, Any] = {str(k): v for k, v in meta.items()}  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
    return SkillMetadata(
        name=str(meta_typed["name"]),
        description=str(meta_typed.get("description", "")),
        path=str(path),
        tags=meta_typed.get("tags", []),
        allowed_tools=meta_typed.get("allowed-tools", []),
    )


def _get_body(path: Path) -> str:
    """Return the SKILL.md content below the YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match:
        return text[match.end() :].strip()
    return text.strip()


class SkillRegistry:
    """Loads and indexes skills from one or more directories.

    Each skill is a subdirectory containing a ``SKILL.md`` file with YAML
    frontmatter.  Later directories win for same-named skills.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skills: dict[str, SkillMetadata] = {}

    def load_from_directory(self, directory: str | Path) -> int:
        """Scan *directory* for skill subdirectories.  Returns count loaded."""
        root = Path(directory)
        if not root.is_dir():
            logger.warning("Skill directory does not exist: %s", root)
            return 0

        count = 0
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.is_file():
                meta = _parse_skill_md(skill_md)
                if meta is not None:
                    self._skills[meta.name] = meta
                    count += 1
        return count

    # -- Lookup --

    def get(self, name: str) -> SkillMetadata | None:
        return self._skills.get(name)

    def list_all(self) -> list[SkillMetadata]:
        return list(self._skills.values())

    def get_full_content(self, name: str) -> str | None:
        """Return the body of a SKILL.md (everything below frontmatter)."""
        meta = self._skills.get(name)
        if meta is None:
            return None
        return _get_body(Path(meta.path))

