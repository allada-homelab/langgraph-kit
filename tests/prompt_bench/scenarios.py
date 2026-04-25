"""Scenario file format and loader.

A *scenario* pairs a deterministic input (one or more user turns), an
agent profile, an optional seeded state, expected behaviour assertions,
and a rubric reference. Scenarios live as YAML files under
``tests/prompt_bench/scenarios/<target>/<id>.yaml``.

Example scenario::

    id: multi_turn_memory_recall
    target: reference_deep_agent.core_identity
    description: |
      3-turn conversation: user states a fact, asks an unrelated
      question, then asks back about the original fact. Agent must
      recall via memory.
    agent_profile: reference_deep_agent
    seeded_state:
      memory: []
      workspace: tmp
    turns:
      - user: "My favorite color is teal."
      - user: "What's 2+2?"
      - user: "What did I tell you my favorite color was?"
    expected_behaviors:
      must_call_tool: ["memory_save", "memory_search"]
      must_mention: ["teal"]
      must_not_mention: ["I don't know", "I don't recall"]
    rubric: rubrics/memory_recall_quality.md
    samples: 5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path


class ScenarioTurn(BaseModel):
    """One turn in a scenario conversation."""

    model_config = ConfigDict(extra="forbid")

    user: str


class ExpectedBehavior(BaseModel):
    """Structural assertions a scenario expects from the agent's output.

    None of these are pass/fail gates by themselves — they feed into
    rule-based metrics that contribute to the overall regression check.
    """

    model_config = ConfigDict(extra="forbid")

    must_call_tool: list[str] = Field(default_factory=list)
    must_not_call_tool: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    min_tool_calls: int | None = None


class Scenario(BaseModel):
    """A single benchmark scenario."""

    model_config = ConfigDict(extra="forbid")

    id: str
    target: str
    description: str = ""
    agent_profile: str = "reference_deep_agent"
    seeded_state: dict[str, Any] = Field(default_factory=dict)
    turns: list[ScenarioTurn]
    expected_behaviors: ExpectedBehavior = Field(default_factory=ExpectedBehavior)
    rubric: str | None = None
    samples: int = 5

    @field_validator("turns")
    @classmethod
    def _at_least_one_turn(cls, v: list[ScenarioTurn]) -> list[ScenarioTurn]:
        if not v:
            msg = "Scenario must declare at least one user turn"
            raise ValueError(msg)
        return v

    @field_validator("samples")
    @classmethod
    def _positive_samples(cls, v: int) -> int:
        if v < 1:
            msg = "samples must be >= 1"
            raise ValueError(msg)
        return v


def load_scenario(path: Path) -> Scenario:
    """Load a single scenario YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Scenario file {path} must contain a YAML mapping at top level"
        raise ValueError(msg)
    return Scenario.model_validate(raw)


def discover_scenarios(
    root: Path, target: str | None = None
) -> list[Scenario]:
    """Discover scenarios under ``root/scenarios/``.

    If *target* is provided, only scenarios under
    ``root/scenarios/<target>/`` are loaded. Otherwise every scenario
    under any subdirectory is loaded.
    """
    base = root / "scenarios"
    if not base.is_dir():
        return []

    if target is not None:
        # Allow either the bare target name or the dotted target
        # (e.g. ``reference_deep_agent.core_identity``) — both map to
        # the same directory in practice since the directory groups
        # by agent/target name.
        target_dir = base / target.replace(".", "_")
        if not target_dir.is_dir():
            target_dir = base / target.split(".", 1)[0]
        if not target_dir.is_dir():
            return []
        candidates = sorted(target_dir.glob("*.yaml"))
    else:
        candidates = sorted(base.rglob("*.yaml"))

    return [load_scenario(p) for p in candidates]
