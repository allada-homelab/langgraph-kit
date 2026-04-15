"""Evaluation runner — fetches Langfuse traces, applies metrics, posts scores."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph_kit.evals.models import (
    EvalMetric,
    EvalReport,
    MetricSummary,
    TraceData,
)

logger = logging.getLogger(__name__)

# Threshold for counting a numeric score as "passing"
DEFAULT_PASS_THRESHOLD = 0.7


class EvalRunner:
    """Fetches traces from Langfuse, scores them, and posts results back."""

    def __init__(
        self,
        langfuse: Any,
        metrics: list[EvalMetric],
        llm: Any | None = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        self._langfuse = langfuse
        self._metrics = metrics
        self._llm = llm
        self._pass_threshold = pass_threshold

    def _fetch_traces(
        self,
        hours_back: int = 24,
        limit: int = 100,
        tags: list[str] | None = None,
    ) -> list[TraceData]:
        """Fetch recent traces from Langfuse."""
        from_ts = datetime.now(UTC) - timedelta(hours=hours_back)

        kwargs: dict[str, Any] = {
            "limit": limit,
            "from_timestamp": from_ts,
        }
        if tags:
            kwargs["tags"] = tags

        try:
            result = self._langfuse.api.trace.list(**kwargs)
            traces = result.data if hasattr(result, "data") else result
        except Exception:
            logger.exception("Failed to fetch traces from Langfuse")
            return []

        return [
            TraceData(
                id=t.id,
                name=getattr(t, "name", None),
                input=getattr(t, "input", None),
                output=getattr(t, "output", None),
                tags=getattr(t, "tags", []) or [],
                duration_ms=getattr(t, "latency", None),
                metadata=getattr(t, "metadata", {}) or {},
            )
            for t in traces
        ]

    async def run(
        self,
        hours_back: int = 24,
        limit: int = 100,
        tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> EvalReport:
        """Run all metrics against recent traces.

        Args:
            hours_back: How far back to fetch traces
            limit: Maximum number of traces to evaluate
            tags: Optional tag filter for traces
            dry_run: If True, compute scores but don't post to Langfuse
        """
        start_time = time.monotonic()
        traces = self._fetch_traces(hours_back, limit, tags)
        logger.info("Fetched %d traces for evaluation", len(traces))

        report = EvalReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_traces=len(traces),
        )

        # Initialize metric summaries
        for metric in self._metrics:
            report.metrics[metric.name] = MetricSummary(
                name=metric.name,
                data_type=metric.data_type,
            )

        # Score each trace
        for trace in traces:
            trace_result: dict[str, Any] = {"trace_id": trace.id, "name": trace.name}

            for metric in self._metrics:
                try:
                    result = await metric.score(trace)
                except Exception:
                    logger.exception(
                        "Metric '%s' failed on trace %s", metric.name, trace.id
                    )
                    continue

                trace_result[metric.name] = {
                    "value": result.value,
                    "comment": result.comment,
                }

                # Update summary
                summary = report.metrics[metric.name]
                summary.count += 1
                summary.values.append(result.value)

                # Post score to Langfuse
                if not dry_run:
                    try:
                        self._langfuse.create_score(
                            trace_id=trace.id,
                            name=metric.name,
                            value=result.value,
                            comment=result.comment or "",
                            data_type=metric.data_type,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to post score for metric '%s' on trace %s",
                            metric.name,
                            trace.id,
                        )

            report.trace_results.append(trace_result)

        # Compute aggregates
        for summary in report.metrics.values():
            numeric_vals = [v for v in summary.values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric_vals:
                summary.mean = round(sum(numeric_vals) / len(numeric_vals), 3)
                summary.pass_rate = round(
                    sum(1 for v in numeric_vals if v >= self._pass_threshold)
                    / len(numeric_vals),
                    3,
                )
            bool_vals = [v for v in summary.values if isinstance(v, bool)]
            if bool_vals:
                summary.pass_rate = round(
                    sum(1 for v in bool_vals if v) / len(bool_vals), 3
                )

        report.duration_seconds = round(time.monotonic() - start_time, 2)

        if not dry_run:
            try:
                self._langfuse.flush()
            except Exception:
                logger.debug("Langfuse flush failed", exc_info=True)

        return report
