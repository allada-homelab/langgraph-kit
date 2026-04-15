"""HITL models matching the LangGraph HumanInterrupt schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    """Describes the action the agent wants to perform."""

    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class HumanInterruptConfig(BaseModel):
    """Controls which response actions the UI should offer."""

    allow_ignore: bool = True
    allow_respond: bool = True
    allow_edit: bool = False
    allow_accept: bool = True


class HumanInterrupt(BaseModel):
    """Payload sent to the frontend when the agent calls interrupt()."""

    action_request: ActionRequest
    config: HumanInterruptConfig = HumanInterruptConfig()
    description: str | None = None


class HumanResponse(BaseModel):
    """User's response to an interrupt, sent back via the resume endpoint."""

    type: Literal["accept", "ignore", "response", "edit"]
    args: str | dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    """Request body for the resume endpoint."""

    responses: list[HumanResponse]


class ThreadStateResponse(BaseModel):
    """Response from the thread state endpoint."""

    thread_id: str
    status: Literal["idle", "interrupted", "busy", "error"]
    interrupts: list[dict[str, Any]] = []
