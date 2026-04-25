"""Models for the audit log.

Append-only tuples ``(timestamp, actor, action, target, metadata)``
describing who did what, to what, when. Stored via
:class:`langgraph_kit.core.audit.AuditStore` in time-bucketed
namespaces so listings stay cheap.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AuditAction(StrEnum):
    """Bounded set of audit-worthy action kinds.

    Extend conservatively — every consumer that filters on action
    must be updated when new variants land.
    """

    AGENT_INVOKE = "agent_invoke"
    AGENT_RUN_COMPLETE = "agent_run_complete"
    MEMORY_CREATE = "memory_create"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    HITL_APPROVE = "hitl_approve"
    HITL_REJECT = "hitl_reject"
    INJECTION_DETECTED = "injection_detected"
    OUTPUT_REDACTED = "output_redacted"
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"


class AuditEntry(BaseModel):
    """A single immutable audit row."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Actor: ``user:<id>``, ``agent:<id>``, ``system``, or any caller-defined
    # prefix. Free-form on purpose so audit consumers don't need to mirror
    # an enum that grows over time.
    actor: str
    action: AuditAction
    # Target: the entity acted upon. ``thread:<id>``, ``memory:<id>``,
    # ``message:<id>``, ``namespace:<tuple>``. Free-form like ``actor``.
    target: str
    # Free-form payload. The schema is intentionally flexible because
    # audit producers vary widely (a memory delete carries the deleted
    # title; an injection detection carries pattern names).
    metadata: dict[str, Any] = Field(default_factory=dict)

    def bucket_key(self) -> str:
        """Year-month bucket the entry lives under in the Store."""
        return f"{self.timestamp.year:04d}_{self.timestamp.month:02d}"
