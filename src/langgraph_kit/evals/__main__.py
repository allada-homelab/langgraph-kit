"""CLI entry point: ``python -m langgraph_kit.evals``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Available LLM-graded rubrics (must have a matching prompt file in metrics/prompts/).
# Keep in sync with the ``metrics/prompts/*.md`` files on disk — previously
# only three of the five shipped rubrics were wired here, so ``safety``
# and ``tool_efficiency`` never ran from the CLI.
_LLM_RUBRICS = [
    "faithfulness",
    "helpfulness",
    "safety",
    "task_completion",
    "tool_efficiency",
]


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
    # --- CI integration flags (issue #14) ---
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the overall pass rate (mean across "
            "metrics that report one) is below this threshold. "
            "Range: 0.0 to 1.0."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Path to a previous --ci-json output to compare against. "
            "Exits non-zero when this run's pass_rate drops by more "
            "than --baseline-tolerance vs the baseline's pass_rate."
        ),
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.0,
        help=(
            "Allowed pass_rate drop vs the baseline before failing "
            "(default: 0.0 = any regression fails)."
        ),
    )
    parser.add_argument(
        "--ci-json",
        type=Path,
        default=None,
        help=(
            "Write a slim, schema-versioned JSON report to this path. "
            "Suitable for CI artifact upload and as input to a future "
            "--baseline run. Distinct from --report-file which writes "
            "the full EvalReport."
        ),
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()

    # --- Initialize Langfuse ---
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.error("langfuse package not installed. Run: uv sync")
        return 1

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
            return 1

    logger.info(
        "Running %d metric(s): %s",
        len(all_metrics),
        ", ".join(m.name for m in all_metrics),
    )

    # --- Run ---
    from langgraph_kit.evals.report import (
        check_ci_thresholds,
        print_console_report,
        report_to_ci_json,
        write_json_report,
    )
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

    # --- CI integration: slim JSON + thresholds + baseline ---
    if args.ci_json is not None:
        ci_payload = report_to_ci_json(report)
        args.ci_json.parent.mkdir(parents=True, exist_ok=True)
        args.ci_json.write_text(json.dumps(ci_payload, indent=2), encoding="utf-8")
        logger.info("CI JSON written to %s", args.ci_json)

    baseline_payload: dict[str, Any] | None = None
    if args.baseline is not None:
        try:
            baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.error("Baseline file not found: %s", args.baseline)
            return 1
        except json.JSONDecodeError as e:
            logger.error("Baseline file is not valid JSON: %s (%s)", args.baseline, e)
            return 1

    failures = check_ci_thresholds(
        report,
        fail_under=args.fail_under,
        baseline=baseline_payload,
        baseline_tolerance=args.baseline_tolerance,
    )
    if failures:
        for reason in failures:
            sys.stderr.write(f"FAIL: {reason}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
