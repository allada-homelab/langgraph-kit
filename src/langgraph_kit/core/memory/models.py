"""Typed memory models with explicit taxonomy and scoping."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """What kind of knowledge a memory record represents."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class MemoryScope(StrEnum):
    """Visibility boundary for a memory record."""

    USER = "user"
    ASSISTANT = "assistant"
    PROJECT = "project"
    TEAM = "team"


class MemoryRecord(BaseModel):
    """A single unit of persistent memory stored in LangGraph Store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    type: MemoryType
    scope: MemoryScope
    summary: str
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str | None = None

    def to_store_value(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LangGraph Store `.aput()`."""
        return self.model_dump(mode="json")

    @classmethod
    def from_store_value(cls, data: dict[str, Any]) -> MemoryRecord:
        """Deserialize from a LangGraph Store item value dict."""
        return cls.model_validate(data)
