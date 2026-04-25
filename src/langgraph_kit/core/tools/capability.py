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

    Several fields are now *enforced* by bundled middleware:

    - ``max_output_chars`` — honored by
      :class:`ResultPersistenceMiddleware`; per-tool override of the
      middleware's default character threshold.
    - ``offload_large_results`` — honored by
      :class:`ResultPersistenceMiddleware`; ``False`` opts the tool out
      of persistence regardless of size.
    - ``interrupt_before`` — honored by
      :class:`AutoInterruptMiddleware`; ``True`` triggers HITL gating
      before the tool runs.

    ``max_output_tokens`` is retained as an advisory hint for callers
    layering tokenizer-aware middleware (the bundled middleware caps in
    characters to stay provider-agnostic, see ``max_output_chars``).
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
    # Advisory hint for caller-supplied tokenizer-aware middleware.
    max_output_tokens: int | None = None
    # Enforced by ResultPersistenceMiddleware: per-tool override of the
    # character threshold for "large enough to persist". None = use the
    # middleware's default.
    max_output_chars: int | None = None
    # Enforced by ResultPersistenceMiddleware. True (default) = persist
    # this tool's results when over threshold; False = always inline,
    # regardless of size. Default flipped from False to True in 1.1: the
    # previous default was a no-op (the field was advisory) so callers
    # who set it to False explicitly now get the opt-out behaviour they
    # were trying to express.
    offload_large_results: bool = True
    # Enforced by AutoInterruptMiddleware. True = HITL approval prompt
    # before the tool runs.
    interrupt_before: bool = False
