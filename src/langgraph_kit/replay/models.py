"""Data models for conversation recording and replay."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMInteraction(BaseModel):
    """A single LLM call/response pair."""

    kind: Literal["llm"] = "llm"
    sequence_num: int
    model: str = ""
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    output_message: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] | None = None
    timestamp: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolInteraction(BaseModel):
    """A single tool call/response pair."""

    kind: Literal["tool"] = "tool"
    sequence_num: int
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str = ""
    status: Literal["success", "error"] = "success"
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationRecording(BaseModel):
    """A complete recorded conversation with all LLM and tool interactions."""

    id: str = ""
    agent_id: str = ""
    thread_id: str = ""
    model: str = ""
    created_at: str = ""
    interactions: list[LLMInteraction | ToolInteraction] = Field(default_factory=list)
    user_messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def llm_interactions(self) -> list[LLMInteraction]:
        """Return only LLM interactions in order."""
        return [i for i in self.interactions if isinstance(i, LLMInteraction)]

    @property
    def tool_interactions(self) -> list[ToolInteraction]:
        """Return only tool interactions in order."""
        return [i for i in self.interactions if isinstance(i, ToolInteraction)]

    @property
    def tool_sequence(self) -> list[str]:
        """Return the ordered list of tool names called."""
        return [i.tool_name for i in self.tool_interactions]
