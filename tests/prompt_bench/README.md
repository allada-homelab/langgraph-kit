# prompt_bench — internal prompt-optimization harness

Internal tooling for systematically optimizing the prompts the kit ships
(agent system prompts, prompt-assembly sections, middleware prompts,
worker definitions, skill content). **Not** part of the public API —
lives under `tests/` deliberately.

## Why pairwise, not absolute scoring

Absolute scores from a single LLM judge are noisy and drift across
models, runs, and prompt families. Pairwise comparisons (A vs B for
the same input, with the same scenario state, with the same execution
LLM) are far more stable: the judge only has to decide "which is
better and why", not "score this 0-1." The literature converges on
pairwise preference as the lower-variance signal, and this harness
follows that convention.

## Acceptance bar (strict)

A prompt change ships only if **all** hold:

1. **Pairwise win rate ≥ 60%** of decided pairs across all scenarios
   for the target prompt.
2. **Two-judge agreement ≥ 70%** across all pairs (pairs are *decided*
   only when every judge agrees on a non-tie winner *or* every judge
   agrees on a tie).
3. **No rule-based metric regression > 5%** on `LatencyMetric`,
   `ErrorFreeMetric`, `ToolEfficiencyMetric`, `ResponseLengthMetric`.
4. **Cross-prompt regression suite passes** — small bench against
   scenarios from other tiers to catch coupling effects.
5. **N=5 samples per scenario**; high-variance scenarios (IQR > 0.3
   on score) get re-run with N=10 before counting.

The thresholds are encoded in `diff.py`:
`WIN_RATE_THRESHOLD`, `JUDGE_AGREEMENT_THRESHOLD`,
`METRIC_REGRESSION_TOLERANCE`. Adjust there if the bar moves; do not
override per-PR.

## Layout

```
tests/prompt_bench/
├── runner.py        BenchRunner — iterates scenarios x samples
├── pairwise.py      PairwiseJudge + PairwisePanel — A/B comparison
├── variants.py      PromptOverlay + middleware patch context manager
├── profiles.py      Per-agent overlay-application bridges
├── diff.py          DiffReport + acceptance-criteria evaluation
├── scenarios.py     Scenario YAML schema + loader
├── conftest.py      Pytest fixtures (mocked LLM by default; real LLM
│                    when PROMPT_BENCH_LLM=real and AGENT_LLM_API_KEY)
├── run.py           CLI entry: list-targets, list-scenarios, ...
├── scenarios/<target>/*.yaml      Scenario library, grows over time
├── rubrics/*.md                   Pairwise + per-target rubrics
└── variants/<target>/*.md         baseline.md + named candidates
```

## Variant dimensions to try (one per iteration)

Each candidate variant should change exactly one dimension so we know
what moved the needle. Pulled from current Anthropic prompt-engineering
guidance:

- **Structure**: XML tags vs markdown headings vs prose paragraphs
- **Role priming**: "You are X" vs imperative vs second-person
- **Examples**: 0-shot vs 1-shot vs few-shot; positive only vs positive+negative
- **Specificity**: terse rules vs verbose explanations with rationale
- **Output format**: explicit JSON schema vs natural language
- **Position**: instructions in system message vs first user message vs both
- **Ordering**: instructions-then-context vs context-then-instructions

## Workflow — adding a new variant

1. **Pick a target.** `python -m tests.prompt_bench.run list-targets`
2. **Author the candidate.** Drop a markdown file under
   `variants/<target>/<variant_name>.md`. The file may carry a YAML
   frontmatter block (notes, rationale, dimension changed) — the
   loader strips it. Everything below the frontmatter is the prompt
   text the overlay will inject.
3. **Run the bench.** Once the live `run`/`diff` CLI lands (Phase 1),
   it'll be:

   ```bash
   PROMPT_BENCH_LLM=real AGENT_LLM_API_KEY=... \
     python -m tests.prompt_bench.run run --target reference_deep_agent.core_identity --variant my_candidate
   ```

   For now the bench is exercised via the pytest harness and the
   nightly workflow.
4. **Diff.** `python -m tests.prompt_bench.run diff --base base.json --variant variant.json --out diff.md`
5. **Open a PR** with the variant file, the diff report (markdown),
   and a one-paragraph note on which dimension you changed.

## Workflow — adding a new scenario

A scenario is a YAML file under `scenarios/<target>/`. Schema lives in
`scenarios.py:Scenario`. Required fields:

- `id` — unique identifier
- `target` — dotted prompt target (e.g. `reference_deep_agent.core_identity`)
- `turns` — list of user turns
- `samples` — N for the harness (default 5)
- `rubric` — relative path to a rubric markdown file (per-target rubric
  for the LLM judge; the pairwise system rubric is added on top)

Optional: `seeded_state`, `expected_behaviors`, `description`.

## Hermetic vs real-LLM

Default mode is **hermetic** — `bench_llm` is a deterministic stub.
Use this for:

- Unit-testing harness logic (`pytest tests/prompt_bench/`)
- CI's per-PR run (no API key required)

**Real-LLM mode** is enabled by `PROMPT_BENCH_LLM=real` +
`AGENT_LLM_API_KEY`. Use this for:

- Actual prompt-iteration cycles
- The nightly CI workflow
- Local validation before opening a PR

The execution LLM defaults to `claude-sonnet-4-6` (quality matters);
judges default to `claude-opus-4-7` and `claude-haiku-4-5`. Override
in `conftest.py` if needed.

## Signal validation (run before optimizing real prompts)

Two sanity checks the harness must pass before any optimization PR is
trusted:

1. **Floor:** baseline-vs-baseline produces a pairwise win rate
   within `[0.45, 0.55]` (noise band). Outside this band → harness
   has bias.
2. **Ceiling:** baseline-vs-`deliberately_broken.md` produces a
   baseline win rate ≥ 80%. Lower → the judges aren't discriminating.

Both are exercised by the nightly workflow. The `deliberately_broken.md`
variants under `variants/<target>/` exist for this reason — do not
delete them.
