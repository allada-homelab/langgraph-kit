"""CLI entry point for the prompt-bench harness.

Usage::

    python -m tests.prompt_bench.run list-targets
    python -m tests.prompt_bench.run list-scenarios [--target <name>]
    python -m tests.prompt_bench.run run --target <name> --variant <variant_name> [--samples N] [--out report.json]
    python -m tests.prompt_bench.run diff --base base.json --variant variant.json [--out diff.md]
    python -m tests.prompt_bench.run signal-check --target <name>

Hermetic by default — uses a deterministic stub LLM so the harness
loop runs end-to-end without a real model. The bench uses an
**OpenAI-compatible** chat endpoint when these are all set::

    LLM_BASE_URL    e.g. http://10.69.1.169:8690/v1 (any OpenAI-compat proxy)
    LLM_API_KEY     api key for the proxy (often a placeholder for local)
    LLM_MODEL       default model name routed through the proxy

Optional per-role overrides — useful when judges need to be different
models (or different families) than the executor::

    BENCH_EXECUTOR_MODEL    model under evaluation     (defaults to LLM_MODEL)
    BENCH_JUDGE_MODEL_A     primary pairwise judge     (defaults to LLM_MODEL)
    BENCH_JUDGE_MODEL_B     secondary pairwise judge   (defaults to LLM_MODEL)

If any of the three required vars (``LLM_BASE_URL``, ``LLM_API_KEY``,
``LLM_MODEL``) is missing, the harness falls back to its deterministic
stub. Stub-mode signal numbers have no meaning beyond "did the loop
crash?". ``signal-check`` refuses to run in stub mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

from tests.prompt_bench.diff import (
    JUDGE_AGREEMENT_THRESHOLD,
    WIN_RATE_THRESHOLD,
    compute_diff,
    render_markdown,
    to_dict,
)
from tests.prompt_bench.live_runner import make_run_one
from tests.prompt_bench.pairwise import PairwiseJudge, PairwisePanel
from tests.prompt_bench.runner import BenchReport, BenchRunner, BenchSample
from tests.prompt_bench.scenarios import discover_scenarios
from tests.prompt_bench.variants import (
    PromptOverlay,
    discover_variants,
    load_variant,
    overlay_from_variant_file,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent

# OpenAI-compatible chat endpoint (any proxy that speaks ``/v1/chat/completions``).
# All three must be set for real-LLM mode; otherwise the harness uses its stub.
_BASE_URL_ENV = "LLM_BASE_URL"
_API_KEY_ENV = "LLM_API_KEY"
_MODEL_ENV = "LLM_MODEL"

# Optional per-role overrides — when a single proxy can route to multiple
# upstream models, pin different ones for executor / judges. If unset,
# every role uses ``LLM_MODEL``.
_EXECUTOR_MODEL_ENV = "BENCH_EXECUTOR_MODEL"
_JUDGE_A_MODEL_ENV = "BENCH_JUDGE_MODEL_A"
_JUDGE_B_MODEL_ENV = "BENCH_JUDGE_MODEL_B"


# ---------------------------------------------------------------------------
# Target registry — declared targets the bench knows about.
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


# Signal-validation thresholds. Floor band is symmetric around 0.5 noise.
SIGNAL_FLOOR_LOW = 0.45
SIGNAL_FLOOR_HIGH = 0.55
SIGNAL_CEILING_MIN = 0.80


# Exit codes
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAIL = 1


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
    p_signal.add_argument("--samples", type=int, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.cmd == "list-targets":
        return _cmd_list_targets()
    if args.cmd == "list-scenarios":
        return _cmd_list_scenarios(args.target)
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args.target, args.variant, args.samples, args.out))
    if args.cmd == "diff":
        return asyncio.run(_cmd_diff(args.base, args.variant, args.out))
    if args.cmd == "signal-check":
        return asyncio.run(_cmd_signal_check(args.target, args.samples))

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


# ---------------------------------------------------------------------------
# `run`
# ---------------------------------------------------------------------------


async def _cmd_run(
    target: str,
    variant_name: str,
    samples_override: int | None,
    out_path: Path | None,
) -> int:
    if target not in TARGETS:
        sys.stderr.write(
            f"Unknown target {target!r}; run list-targets to see valid names\n"
        )
        return EXIT_USAGE

    target_meta = TARGETS[target]
    variants = discover_variants(ROOT, target)
    if variant_name not in variants:
        sys.stderr.write(
            f"No variant named {variant_name!r} under variants/{target}/\n"
            f"Discovered: {sorted(variants)}\n"
        )
        return EXIT_USAGE

    overlay = _build_overlay(target_meta, variant_name, variants[variant_name])
    scenarios = discover_scenarios(ROOT, target=target)
    if samples_override is not None:
        scenarios = [
            s.model_copy(update={"samples": samples_override}) for s in scenarios
        ]
    if not scenarios:
        sys.stderr.write(f"No scenarios found for target {target!r}\n")
        return EXIT_USAGE

    sys.stdout.write(
        f"Running overlay {variant_name!r} against {len(scenarios)} scenarios "
        f"({sum(s.samples for s in scenarios)} total samples).\n"
    )

    report = await _execute_overlay(target_meta["agent"], scenarios, overlay)
    sys.stdout.write(
        f"Done. {len(report.samples)} samples; "
        f"{sum(1 for s in report.samples if s.error)} errors.\n"
    )

    if out_path is not None:
        out_path.write_text(_serialize_report(report), encoding="utf-8")
        sys.stdout.write(f"Wrote {out_path}\n")
    return EXIT_OK


# ---------------------------------------------------------------------------
# `diff`
# ---------------------------------------------------------------------------


async def _cmd_diff(base_path: Path, variant_path: Path, out_path: Path | None) -> int:
    base_report = _deserialize_report(base_path)
    variant_report = _deserialize_report(variant_path)
    sys.stdout.write(
        f"Loaded base={base_report.overlay_name} ({len(base_report.samples)} samples), "
        f"variant={variant_report.overlay_name} ({len(variant_report.samples)} samples).\n"
    )

    panel = _build_pairwise_panel()
    diff = await compute_diff(base_report, variant_report, panel)
    md = render_markdown(diff)
    sys.stdout.write(md)

    if out_path is not None:
        out_path.write_text(md, encoding="utf-8")
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(to_dict(diff), indent=2), encoding="utf-8")
        sys.stdout.write(f"Wrote {out_path} and {json_path}\n")
    return EXIT_OK if diff.passes_acceptance else EXIT_FAIL


# ---------------------------------------------------------------------------
# `signal-check` — runs baseline-vs-baseline + baseline-vs-deliberately-broken.
# ---------------------------------------------------------------------------


async def _cmd_signal_check(target: str, samples_override: int | None) -> int:
    if target not in TARGETS:
        sys.stderr.write(f"Unknown target {target!r}\n")
        return EXIT_USAGE

    if not _real_llm_enabled():
        sys.stderr.write(
            "signal-check requires a real LLM. Set all three of "
            f"{_BASE_URL_ENV}, {_API_KEY_ENV}, {_MODEL_ENV} before running.\n"
            f"Current: {_BASE_URL_ENV}={'set' if os.environ.get(_BASE_URL_ENV) else '<unset>'}, "
            f"{_API_KEY_ENV}={'set' if os.environ.get(_API_KEY_ENV) else '<unset>'}, "
            f"{_MODEL_ENV}={os.environ.get(_MODEL_ENV, '<unset>')!r}\n"
        )
        return EXIT_USAGE

    target_meta = TARGETS[target]
    variants = discover_variants(ROOT, target)
    for required in ("baseline", "deliberately_broken"):
        if required not in variants:
            sys.stderr.write(
                f"Missing required variant {required!r} for target {target!r}; "
                f"both baseline and deliberately_broken must exist for signal-check.\n"
            )
            return EXIT_USAGE

    scenarios = discover_scenarios(ROOT, target=target)
    if samples_override is not None:
        scenarios = [
            s.model_copy(update={"samples": samples_override}) for s in scenarios
        ]
    if not scenarios:
        sys.stderr.write(f"No scenarios for target {target!r}\n")
        return EXIT_USAGE

    sys.stdout.write(
        f"signal-check for {target!r} ({len(scenarios)} scenarios, "
        f"{sum(s.samples for s in scenarios)} samples per overlay)\n"
    )

    base_overlay = _build_overlay(target_meta, "baseline", variants["baseline"])
    # For the ceiling check we want a catastrophically broken agent. A
    # single-section override gets drowned out by the ~9 other sections
    # the kit assembles (memory, orchestration, ACTIVATION_SECTIONS,
    # tool_guidance, etc.) — the agent stays helpful and judges call ties.
    # Apply the broken text to every core section the profile ships so
    # the agent has nothing helpful to fall back on. Phase-1+ runs use
    # ``run`` (single-section overlay) where weak signal IS what we want
    # to measure; signal-check's job is to prove the harness can detect
    # a clear difference, so we bias toward a strong one.
    broken_overlay = _build_ceiling_overlay(
        target_meta, variants["deliberately_broken"]
    )

    # Run baseline twice (independent samples) to measure the noise floor,
    # plus the deliberately-broken variant for the ceiling check.
    base_a = await _execute_overlay(
        target_meta["agent"], scenarios, base_overlay, suffix="_a"
    )
    base_b = await _execute_overlay(
        target_meta["agent"], scenarios, base_overlay, suffix="_b"
    )
    broken = await _execute_overlay(target_meta["agent"], scenarios, broken_overlay)

    panel = _build_pairwise_panel()
    floor = await compute_diff(base_a, base_b, panel)
    ceiling_diff = await compute_diff(base_a, broken, panel)
    # ``compute_diff`` reports ``overall_win_rate`` as the *variant* win rate.
    # For the ceiling check we want the *baseline* win rate (i.e. how often
    # baseline beat the broken variant).
    ceiling_baseline_win_rate = (
        ceiling_diff.total_base_wins / ceiling_diff.total_decided
        if ceiling_diff.total_decided
        else 0.0
    )
    # Tie-rate matters: when prompts are identical (the floor case),
    # judges *should* call most pairs "tie". A 0% variant-win rate
    # paired with a high tie rate is healthy; the same 0% paired with
    # all base_wins would be position bias. The non-tie split tells
    # them apart.
    floor_non_tie = floor.total_variant_wins + floor.total_base_wins
    floor_non_tie_variant_share = (
        floor.total_variant_wins / floor_non_tie if floor_non_tie else 0.5
    )

    sys.stdout.write("\n=== Floor (baseline vs baseline) ===\n")
    sys.stdout.write(
        f"win_rate={floor.overall_win_rate:.3f} "
        f"agreement={floor.overall_judge_agreement:.3f} "
        f"({floor.total_decided}/{floor.total_pairs} decided)\n"
    )
    sys.stdout.write(
        f"  breakdown: ties={floor.total_ties} "
        f"base_wins={floor.total_base_wins} variant_wins={floor.total_variant_wins} "
        f"undecided={floor.total_pairs - floor.total_decided}\n"
    )
    sys.stdout.write(
        f"  non-tie variant share: {floor_non_tie_variant_share:.3f} "
        f"({floor.total_variant_wins}/{floor_non_tie}) — "
        f"want close to 0.5 for unbiased judges\n"
    )
    # Floor passes when either the legacy band holds *or* the non-tie
    # split is unbiased (both checks tolerate the all-ties case where
    # ``floor_non_tie == 0`` falls back to 0.5).
    floor_ok = (
        SIGNAL_FLOOR_LOW <= floor.overall_win_rate <= SIGNAL_FLOOR_HIGH
        or SIGNAL_FLOOR_LOW <= floor_non_tie_variant_share <= SIGNAL_FLOOR_HIGH
    )
    sys.stdout.write(
        f"  floor: {'OK' if floor_ok else 'FAIL'} (band [{SIGNAL_FLOOR_LOW}, {SIGNAL_FLOOR_HIGH}])\n"
    )

    sys.stdout.write("\n=== Ceiling (baseline vs deliberately_broken) ===\n")
    sys.stdout.write(
        f"baseline_win_rate={ceiling_baseline_win_rate:.3f} "
        f"agreement={ceiling_diff.overall_judge_agreement:.3f} "
        f"({ceiling_diff.total_decided}/{ceiling_diff.total_pairs} decided)\n"
    )
    sys.stdout.write(
        f"  breakdown: base_wins={ceiling_diff.total_base_wins} "
        f"ties={ceiling_diff.total_ties} variant_wins={ceiling_diff.total_variant_wins} "
        f"undecided={ceiling_diff.total_pairs - ceiling_diff.total_decided}\n"
    )
    ceiling_ok = ceiling_baseline_win_rate >= SIGNAL_CEILING_MIN
    sys.stdout.write(
        f"  ceiling: {'OK' if ceiling_ok else 'FAIL'} "
        f"(min baseline win rate {SIGNAL_CEILING_MIN})\n"
    )

    sys.stdout.write("\n=== Judge agreement ===\n")
    agreement_ok = floor.overall_judge_agreement >= JUDGE_AGREEMENT_THRESHOLD
    sys.stdout.write(
        f"  floor agreement: {floor.overall_judge_agreement:.3f} — "
        f"{'OK' if agreement_ok else 'FAIL'} "
        f"(min {JUDGE_AGREEMENT_THRESHOLD})\n"
    )

    overall_ok = floor_ok and ceiling_ok and agreement_ok
    sys.stdout.write(f"\nsignal-check: {'PASS' if overall_ok else 'FAIL'}\n")
    return EXIT_OK if overall_ok else EXIT_FAIL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_overlay(
    target_meta: dict[str, str],
    variant_name: str,
    variant_path: Path,
) -> PromptOverlay:
    text = load_variant(variant_path)
    if target_meta["kind"] == "section":
        return overlay_from_variant_file(
            name=variant_name,
            section_id=target_meta["section_id"],
            text=text,
        )
    return overlay_from_variant_file(
        name=variant_name,
        middleware_attr=target_meta["module_attr"],
        text=text,
    )


def _build_ceiling_overlay(
    target_meta: dict[str, str],
    variant_path: Path,
) -> PromptOverlay:
    """Strong-signal broken overlay for signal-check ceiling validation.

    For ``kind=section`` targets, applies the broken text to *every*
    core section the profile ships (not just the named one). A
    single-section override gets drowned out by the other sections the
    kit assembles (memory, orchestration, ACTIVATION_SECTIONS, etc.) —
    the agent stays helpful and judges call ties. Overriding all core
    sections leaves the agent with no helpful instruction to fall back
    on, producing a clear "broken" output the judges can confidently
    rank below baseline.

    For ``kind=middleware`` targets, falls back to the standard
    single-attribute swap (middleware constants don't have the
    multi-section fallback problem).
    """
    text = load_variant(variant_path)
    if target_meta["kind"] != "section":
        return overlay_from_variant_file(
            name="deliberately_broken",
            middleware_attr=target_meta["module_attr"],
            text=text,
        )

    from tests.prompt_bench.profiles import get_baseline_sections

    section_ids = [s.id for s in get_baseline_sections(target_meta["agent"])]
    return PromptOverlay(
        name="deliberately_broken",
        section_overrides=dict.fromkeys(section_ids, text),
    )


async def _execute_overlay(
    profile_name: str,
    scenarios: list[Any],
    overlay: PromptOverlay,
    *,
    suffix: str = "",
) -> BenchReport:
    """Run *scenarios* under *overlay* with the configured executor LLM."""
    executor_llm = _build_executor_llm()
    run_one = make_run_one(profile_name, executor_llm)
    runner = BenchRunner(run_one)
    overlay_for_run = (
        overlay.model_copy(update={"name": overlay.name + suffix})
        if suffix
        else overlay
    )
    return await runner.run(scenarios, overlay_for_run)


def _build_executor_llm() -> Any:
    if _real_llm_enabled():
        return _build_real_chat_model(_role_model(_EXECUTOR_MODEL_ENV))
    return _StubExecutorLLM()


def _build_pairwise_panel() -> PairwisePanel:
    if _real_llm_enabled():
        model_a = _role_model(_JUDGE_A_MODEL_ENV)
        model_b = _role_model(_JUDGE_B_MODEL_ENV)
        judges = [
            PairwiseJudge(
                name=f"judge_a:{model_a}", llm=_build_real_chat_model(model_a)
            ),
            PairwiseJudge(
                name=f"judge_b:{model_b}", llm=_build_real_chat_model(model_b)
            ),
        ]
    else:
        judges = [
            PairwiseJudge(
                name=f"stub-{i}",
                llm=_StubJudgeLLM(default_winner="tie"),
                rubric_text="dummy",
            )
            for i in range(2)
        ]
    return PairwisePanel(judges=judges, rng=random.Random(0))


def _real_llm_enabled() -> bool:
    """Real mode is active iff base URL + key + default model are all set."""
    return all(
        os.environ.get(name) for name in (_BASE_URL_ENV, _API_KEY_ENV, _MODEL_ENV)
    )


def _role_model(role_env: str) -> str:
    """Pick the per-role model override, falling back to ``LLM_MODEL``."""
    return os.environ.get(role_env) or os.environ[_MODEL_ENV]


def _build_real_chat_model(model_name: str) -> Any:
    """Return a ``ChatOpenAI`` pointed at the configured proxy.

    The proxy must speak the OpenAI ``/v1/chat/completions`` shape;
    most local LLM servers (vLLM, llama-server, LiteLLM proxy, etc.)
    do, even when routing to non-OpenAI upstreams.
    """
    from langchain_openai import (  # pyright: ignore[reportMissingImports]
        ChatOpenAI,
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": os.environ[_API_KEY_ENV],
        "base_url": os.environ[_BASE_URL_ENV],
        "max_tokens": 1024,
        "timeout": 60,
    }
    return ChatOpenAI(**kwargs)  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# Stubs — let the harness loop run end-to-end without a real LLM. Numbers
# from stub mode are meaningless beyond "did the wiring crash?".
# ---------------------------------------------------------------------------


class _StubExecutorLLM:
    """Minimal chat-model stub for the executor.

    Returns a single ``AIMessage`` with no tool calls. The deep-agent
    middleware will issue more LLM calls than turns, so this stub is
    designed to handle infinite invocations — every call returns the
    same canned response.
    """

    async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> Any:
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
        )

        return AIMessage(content="[stub response]")

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, **kwargs)
        )

    def with_structured_output(self, _schema: Any) -> Any:
        return self

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
        return self


class _StubJudgeLLM:
    """Always returns ``{"winner": <default>, ...}``."""

    def __init__(self, default_winner: str = "tie") -> None:
        super().__init__()
        self._winner = default_winner

    async def ainvoke(self, _messages: list[Any]) -> Any:
        payload = json.dumps(
            {"winner": self._winner, "confidence": 0.5, "reason": "stub"}
        )

        class _Response:
            content = payload

        return _Response()


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def _serialize_report(report: BenchReport) -> str:
    return json.dumps(
        {
            "overlay_name": report.overlay_name,
            "samples": [
                {
                    "scenario_id": s.scenario_id,
                    "sample_index": s.sample_index,
                    "overlay_name": s.overlay_name,
                    "duration_ms": s.duration_ms,
                    "final_output": s.final_output,
                    "tool_calls": s.tool_calls,
                    "error": s.error,
                    "user_input": s.user_input,
                }
                for s in report.samples
            ],
        },
        indent=2,
    )


def _deserialize_report(path: Path) -> BenchReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchReport(
        overlay_name=payload["overlay_name"],
        samples=[
            BenchSample(
                scenario_id=s["scenario_id"],
                sample_index=s["sample_index"],
                overlay_name=s["overlay_name"],
                duration_ms=s["duration_ms"],
                final_output=s["final_output"],
                tool_calls=s.get("tool_calls", []),
                error=s.get("error"),
                user_input=s.get("user_input", ""),
            )
            for s in payload["samples"]
        ],
    )


# Surface the strict diff acceptance threshold for callers / docs.
__all__ = [
    "JUDGE_AGREEMENT_THRESHOLD",
    "SIGNAL_CEILING_MIN",
    "SIGNAL_FLOOR_HIGH",
    "SIGNAL_FLOOR_LOW",
    "TARGETS",
    "WIN_RATE_THRESHOLD",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
