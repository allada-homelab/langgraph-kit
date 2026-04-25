"""CLI entry point for the prompt-bench harness.

Usage::

    python -m tests.prompt_bench.run list-targets
    python -m tests.prompt_bench.run list-scenarios [--target <name>]
    python -m tests.prompt_bench.run run --target <name> --variant <variant_name> [--samples N] [--out report.json]
    python -m tests.prompt_bench.run diff --base base.json --variant variant.json [--out diff.md]
    python -m tests.prompt_bench.run signal-check --target <name>

Phase 0 wires ``list-targets`` and ``list-scenarios`` end-to-end. The
``run`` / ``diff`` / ``signal-check`` subcommands are surfaced (the
nightly workflow already calls ``signal-check``) but exit non-zero
with a clear message until Phase 1 connects the runner to the profile
builders. This split keeps the CLI shape stable so the workflow YAML
doesn't need to be rewritten when live execution lands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tests.prompt_bench.scenarios import discover_scenarios

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Target registry — declared targets the bench knows about (Phase 0 seed).
# ---------------------------------------------------------------------------

# Each target maps a friendly name to either a section ID (for prompt
# sections) or a module:attr path (for middleware constants). Phase 0
# declares the 4 Tier-1 anchors; subsequent phases extend this map as
# they bench more prompts.
TARGETS: dict[str, dict[str, str]] = {
    "reference_deep_agent.core_identity": {
        "kind": "section",
        "section_id": "core_identity",
        "agent": "reference_deep_agent",
    },
    "reference_deep_agent.memory_instructions": {
        "kind": "section",
        "section_id": "memory_instructions",
        "agent": "reference_deep_agent",
    },
    "memory_extraction.prompt": {
        "kind": "middleware",
        "module_attr": "langgraph_kit.core.memory.extraction:_EXTRACTION_PROMPT",
        "agent": "reference_deep_agent",
    },
    "compaction.full_prompt": {
        "kind": "middleware",
        "module_attr": "langgraph_kit.core.context_management.compaction:_FULL_COMPACTION_PROMPT",
        "agent": "reference_deep_agent",
    },
}


# Exit codes
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_YET_WIRED = 3


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-targets", help="List declared bench targets")

    p_scenarios = sub.add_parser("list-scenarios", help="List discovered scenarios")
    p_scenarios.add_argument("--target", default=None)

    p_run = sub.add_parser("run", help="Run a single overlay against scenarios")
    p_run.add_argument("--target", required=True)
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--samples", type=int, default=None)
    p_run.add_argument("--out", type=Path, default=None)

    p_diff = sub.add_parser("diff", help="Diff two existing run reports")
    p_diff.add_argument("--base", required=True, type=Path)
    p_diff.add_argument("--variant", required=True, type=Path)
    p_diff.add_argument("--out", type=Path, default=None)

    p_signal = sub.add_parser(
        "signal-check", help="baseline vs deliberately-broken sanity check"
    )
    p_signal.add_argument("--target", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.cmd == "list-targets":
        return _cmd_list_targets()
    if args.cmd == "list-scenarios":
        return _cmd_list_scenarios(args.target)
    if args.cmd in ("run", "diff", "signal-check"):
        sys.stderr.write(_NOT_YET_WIRED.format(cmd=args.cmd))
        return EXIT_NOT_YET_WIRED

    parser.error(f"Unknown command: {args.cmd}")
    return EXIT_USAGE


def _cmd_list_targets() -> int:
    sys.stdout.write("Declared bench targets:\n")
    for name, meta in sorted(TARGETS.items()):
        sys.stdout.write(f"  {name:<48s} kind={meta['kind']} agent={meta['agent']}\n")
    return EXIT_OK


def _cmd_list_scenarios(target: str | None) -> int:
    scenarios = discover_scenarios(ROOT, target=target)
    if not scenarios:
        msg = (
            f"No scenarios found for target {target!r}\n"
            if target
            else "No scenarios found\n"
        )
        sys.stdout.write(msg)
        return EXIT_OK
    sys.stdout.write(f"Discovered {len(scenarios)} scenarios:\n")
    for s in scenarios:
        sys.stdout.write(f"  [{s.target}] {s.id} (samples={s.samples})\n")
    return EXIT_OK


_NOT_YET_WIRED = (
    "ERROR: `{cmd}` is not yet wired end-to-end.\n"
    "Phase 0 ships the harness modules (BenchRunner, PairwisePanel, "
    "compute_diff, profiles), the scenario library, and the variant "
    "system. Live `{cmd}` lands when the Phase 1 PR connects "
    "tests.prompt_bench.profiles to the runner via a real LLM. Until "
    "then, drive the harness from pytest tests under "
    "tests/prompt_bench/test_*.py.\n"
)


if __name__ == "__main__":
    sys.exit(main())
