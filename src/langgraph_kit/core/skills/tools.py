"""Skill tools — read_skill and list_skills for progressive disclosure."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.skills.registry import SkillRegistry


def build_skill_tools(registry: SkillRegistry) -> list[Any]:
    """Create agent-callable tools for skill discovery and loading.

    Returns a list of async callables: [list_skills, read_skill].
    """

    async def list_skills() -> str:
        """List all available skills with their names and descriptions.

        Use this to discover what specialized workflows are available.
        """
        skills = registry.list_all()
        if not skills:
            return "No skills are currently registered."

        lines = [f"Found {len(skills)} skill(s):\n"]
        for s in skills:
            desc = s.description.split("\n")[0].strip()
            tags = f" [{', '.join(s.tags)}]" if s.tags else ""
            lines.append(f"- **{s.name}**{tags}: {desc}")
        lines.append(
            "\nCall `read_skill(name)` to load the full instructions for a skill."
        )
        return "\n".join(lines)

    async def read_skill(name: str) -> str:
        """Load the full instructions for a skill by name.

        Call this when a task matches a skill's description to get the
        detailed workflow, constraints, and examples.

        Args:
            name: The skill name (e.g. "code-review", "research")
        """
        content = registry.get_full_content(name)
        if content is None:
            available = [s.name for s in registry.list_all()]
            return (
                f"Skill '{name}' not found. "
                f"Available skills: {', '.join(available) or 'none'}"
            )
        return content

    return [list_skills, read_skill]
