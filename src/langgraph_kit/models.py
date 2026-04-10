"""Pydantic request/response models for agent endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FileAttachment(BaseModel):
    """A file attached to a chat message."""

    name: str
    type: str  # MIME type
    size: int
    data_url: str  # base64 data URL (e.g. "data:image/png;base64,...")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    attachments: list[FileAttachment] = Field(default_factory=list)


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
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"


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


# ---------------------------------------------------------------------------
# Thread management models
# ---------------------------------------------------------------------------


class ThreadMetadataResponse(BaseModel):
    """Thread metadata returned by the API."""

    thread_id: str
    user_id: str
    agent_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message_preview: str = ""
    tags: list[str] = Field(default_factory=list)


class ThreadListResponse(BaseModel):
    threads: list[ThreadMetadataResponse]
    total: int
    limit: int
    offset: int


class ThreadUpdateRequest(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
