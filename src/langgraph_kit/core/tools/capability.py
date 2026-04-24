"""Rich tool capability model with metadata beyond simple callables."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolRisk(StrEnum):
    """Risk level for tool operations."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"


class ToolCapability(BaseModel):
    """Rich tool capability with metadata beyond a simple callable.

    .. note::
        ``max_output_tokens``, ``offload_large_results``, and
        ``interrupt_before`` are **advisory hints** — no kit middleware
        currently enforces them. They're retained for callers that wire
        their own enforcement. ``ResultPersistenceMiddleware`` uses its
        own threshold constants (not ``offload_large_results``) and HITL
        gating is triggered by tool name rather than by
        ``interrupt_before``. If you set one of these fields, you're
        responsible for acting on it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    description: str
    fn: Any  # Callable — typed as Any to avoid requiring stubs for all tool libs
    tags: list[str] = Field(default_factory=list)
    risk: ToolRisk = ToolRisk.READ_ONLY
    prompt_guidance: str | None = None
    profiles: list[str] | None = None
    worker_types: list[str] | None = None
    # Advisory hints — see class docstring. Not enforced by the kit.
    max_output_tokens: int | None = None
    offload_large_results: bool = False
    interrupt_before: bool = False
