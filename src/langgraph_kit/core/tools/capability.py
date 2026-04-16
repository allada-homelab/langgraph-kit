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
    """Rich tool capability with metadata beyond a simple callable."""

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
    max_output_tokens: int | None = None
    offload_large_results: bool = False
    interrupt_before: bool = False
