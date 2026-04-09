"""Pydantic request/response models for agent endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class InvokeRequest(BaseModel):
    messages: list[ChatMessage]
    thread_id: str = ""
    checkpoint_id: str = ""  # for branching: start run from this checkpoint


class InvokeResponse(BaseModel):
    content: str
    thread_id: str


class AgentInfo(BaseModel):
    id: str
    name: str


class AgentListResponse(BaseModel):
    agents: list[AgentInfo]


# ---------------------------------------------------------------------------
# Branching models
# ---------------------------------------------------------------------------


class ForkRequest(BaseModel):
    """Fork a conversation at a specific checkpoint."""

    checkpoint_id: str


class ForkResponse(BaseModel):
    thread_id: str
    new_checkpoint_id: str


class CheckpointInfo(BaseModel):
    checkpoint_id: str | None = None
    parent_checkpoint_id: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    next_nodes: list[str] = Field(default_factory=list)
    message_count: int = 0


# ---------------------------------------------------------------------------
# Queue models
# ---------------------------------------------------------------------------


class QueueMessageRequest(BaseModel):
    """Enqueue a message to a busy thread."""

    content: str
    semantic: Literal["append", "interrupt", "replace_goal"] = "append"
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueueMessageResponse(BaseModel):
    item_id: str
    thread_id: str
    queued: bool
    thread_busy: bool
    queue_depth: int


class QueueStatusResponse(BaseModel):
    thread_id: str
    thread_busy: bool
    queue_depth: int
    items: list[dict[str, Any]]
