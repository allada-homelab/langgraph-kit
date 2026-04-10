"""Data models for execution trace export."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceSpan(BaseModel):
    """A single span in an execution trace."""

    span_id: str = ""
    parent_span_id: str | None = None
    kind: Literal["node", "tool", "llm", "chain"] = "chain"
    name: str = ""
    started_at: str = ""
    ended_at: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[TraceSpan] = Field(default_factory=list)


class TraceRecord(BaseModel):
    """A complete execution trace for a single agent run."""

    trace_id: str = ""
    agent_id: str = ""
    thread_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_ms: float = 0.0
    spans: list[TraceSpan] = Field(default_factory=list)

    @property
    def span_count(self) -> int:
        """Count total spans (including nested children)."""

        def _count(spans: list[TraceSpan]) -> int:
            total = len(spans)
            for s in spans:
                total += _count(s.children)
            return total

        return _count(self.spans)


class TraceSummary(BaseModel):
    """Lightweight summary for trace listing endpoints."""

    trace_id: str
    started_at: str
    duration_ms: float
    span_count: int
