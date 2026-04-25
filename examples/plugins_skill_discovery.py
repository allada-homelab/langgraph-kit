"""Plugins + skills: progressive disclosure via SKILL.md and Python plugins.

What this shows
---------------
- Discovering plugins from a directory via :class:`PluginLoader` —
  each ``.py`` file with a ``contribute()`` function gets loaded and
  its tools merged into a :class:`PluginRegistry`
- Discovering skills from a directory via :class:`SkillRegistry` —
  each subdirectory with a ``SKILL.md`` (YAML frontmatter + body)
  becomes a discoverable skill metadata record
- Reading a skill's full body via :meth:`SkillRegistry.get_full_content`

Both subsystems sit at the kit's "progressive disclosure" boundary:
the agent's base prompt stays small; capability is loaded on demand
when the agent calls ``discover_skills`` or invokes a plugin tool.

The companion plugin used here, ``examples/_sample_plugin.py``, is
shipped alongside this demo. The ``_`` prefix excludes it from the
smoke runner.

How to run
----------
    uv run python -m examples.plugins_skill_discovery

Expected output
---------------
    Loaded 1 plugin contribution(s) (sample-plugin).
    Plugin tools registered: ['sample_lookup']
    Wrote sample SKILL.md to /tmp/lgk-example-XXXX/skills/code-review/SKILL.md
    Discovered 1 skill(s):
      - code-review (Skill that reviews diffs.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples._lib import banner, line, tmp_workspace

_SKILL_BODY = """\
---
name: code-review
description: Skill that reviews diffs.
when_to_use: Trigger on user requests that mention "review my code" or "PR".
---

# Code review skill

When the user asks for a review, walk every changed file and emit
findings in the PASS / WARN / FAIL format the kit's verifier worker
uses.
"""


async def main() -> None:
    banner("plugins_skill_discovery")

    from langgraph_kit.core.plugins.loader import PluginLoader
    from langgraph_kit.core.skills.registry import SkillRegistry

    with tmp_workspace() as workspace:
        # --- Plugins -----------------------------------------------------
        # The sample plugin lives at examples/_sample_plugin.py — pull it
        # into a tempdir under a name without the underscore prefix so
        # PluginLoader scans + loads it (it skips ``_`` files by design).
        plugins_dir = workspace / "plugins"
        plugins_dir.mkdir()
        sample_src = Path(__file__).parent / "_sample_plugin.py"
        (plugins_dir / "sample_plugin.py").write_text(
            sample_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

        loader = PluginLoader()
        contributions = loader.load_from_directory(plugins_dir)
        line(
            f"Loaded {len(contributions)} plugin contribution(s) "
            f"({', '.join(c.plugin_id for c in contributions)})."
        )
        all_tools = [tool.name for c in contributions for tool in c.tools]
        line(f"Plugin tools registered: {all_tools}")

        # --- Skills ------------------------------------------------------
        skills_dir = workspace / "skills"
        review_skill_dir = skills_dir / "code-review"
        review_skill_dir.mkdir(parents=True)
        skill_md = review_skill_dir / "SKILL.md"
        skill_md.write_text(_SKILL_BODY, encoding="utf-8")
        line(f"Wrote sample SKILL.md to {skill_md}")

        registry = SkillRegistry()
        registry.load_from_directory(skills_dir)
        skills = registry.list_all()
        line(f"Discovered {len(skills)} skill(s):")
        for meta in skills:
            line(f"  - {meta.name} ({meta.description})")


if __name__ == "__main__":
    asyncio.run(main())
