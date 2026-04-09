"""Evaluation models — metric base class, results, and report."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceData(BaseModel):
    """Simplified trace data extracted from Langfuse."""

    id: str
    name: str | None = None
    input: dict[str, Any] | str | None = None
    output: dict[str, Any] | str | None = None
    tags: list[str] = Field(default_factory=list)
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of a single metric evaluation on a single trace."""

    value: float | str | bool
    comment: str | None = None


class EvalMetric(ABC):
    """Base class for evaluation metrics."""

    name: str
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]

    @abstractmethod
    async def score(self, trace: TraceData) -> EvalResult:
        """Score a single trace. Returns an EvalResult."""
        ...


class MetricSummary(BaseModel):
    """Aggregated results for a single metric across all traces."""

    name: str
    data_type: str
    count: int = 0
    mean: float | None = None
    pass_rate: float | None = None
    values: list[float | str | bool] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Complete evaluation report."""

    timestamp: str
    model: str = ""
    duration_seconds: float = 0.0
    total_traces: int = 0
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    trace_results: list[dict[str, Any]] = Field(default_factory=list)
