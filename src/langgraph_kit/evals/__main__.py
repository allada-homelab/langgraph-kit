"""CLI entry point: ``python -m langgraph_kit.evals``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Available LLM-graded rubrics (must have a matching prompt file in metrics/prompts/)
_LLM_RUBRICS = ["faithfulness", "helpfulness", "task_completion"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics against Langfuse traces"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How many hours back to fetch traces (default: 24)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum traces to evaluate (default: 100)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Filter traces by tags",
    )
    parser.add_argument(
        "--metric",
        nargs="*",
        help="Run only specific metrics by name (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute scores but don't post to Langfuse",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--no-model-graded",
        action="store_true",
        help="Skip model-graded metrics (rule-based only)",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    # --- Initialize Langfuse ---
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.error("langfuse package not installed. Run: uv sync")
        sys.exit(1)

    langfuse = Langfuse()

    # --- Initialize LLM for model-graded metrics ---
    llm = None
    if not args.no_model_graded:
        try:
            from langgraph_kit.llm import build_llm

            llm = build_llm()
        except Exception:
            logger.warning("Could not initialize LLM — skipping model-graded metrics")

    # --- Assemble metrics ---
    from langgraph_kit.evals.metrics.model_graded import LLMJudgeMetric
    from langgraph_kit.evals.metrics.rule_based import (
        ErrorFreeMetric,
        HasToolCallsMetric,
        LatencyMetric,
        ResponseLengthMetric,
    )
    from langgraph_kit.evals.models import EvalMetric

    all_metrics: list[EvalMetric] = [
        ResponseLengthMetric(),
        HasToolCallsMetric(),
        LatencyMetric(),
        ErrorFreeMetric(),
    ]

    if llm and not args.no_model_graded:
        for rubric_name in _LLM_RUBRICS:
            try:
                all_metrics.append(LLMJudgeMetric(name=rubric_name, llm=llm))
            except FileNotFoundError:
                logger.warning("No rubric found for '%s' — skipping", rubric_name)

    # Filter to requested metrics
    if args.metric:
        all_metrics = [m for m in all_metrics if m.name in args.metric]
        if not all_metrics:
            logger.error("No matching metrics found for: %s", args.metric)
            sys.exit(1)

    logger.info(
        "Running %d metric(s): %s",
        len(all_metrics),
        ", ".join(m.name for m in all_metrics),
    )

    # --- Run ---
    from langgraph_kit.evals.report import print_console_report, write_json_report
    from langgraph_kit.evals.runner import EvalRunner

    runner = EvalRunner(langfuse=langfuse, metrics=all_metrics, llm=llm)
    report = await runner.run(
        hours_back=args.hours,
        limit=args.limit,
        tags=args.tags,
        dry_run=args.dry_run,
    )

    # --- Output ---
    print_console_report(report)

    report_path = args.report_file or (
        f"evals/reports/evaluation_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    if args.report_file or not args.dry_run:
        write_json_report(report, report_path)


if __name__ == "__main__":
    asyncio.run(_main())
