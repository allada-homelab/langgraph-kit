"""Skill metadata model parsed from SKILL.md YAML frontmatter."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class SkillMetadata(BaseModel):
    """Metadata for a discoverable skill, parsed from SKILL.md frontmatter.

    ``allowed_tools`` was removed: the field was never enforced — nothing
    in the kit gated tool access by skill — and its presence misled
    callers into thinking the kit would restrict tool use. If per-skill
    tool restrictions are added back later, they should pair with a
    middleware that actually enforces them.
    """

    name: str
    description: str
    path: str  # filesystem path to SKILL.md
    tags: list[str] = []

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or len(v) > 64:
            msg = "Skill name must be 1-64 characters"
            raise ValueError(msg)
        return v

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return v[:1024]
