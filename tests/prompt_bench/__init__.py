"""Internal prompt-optimization harness.

Lives in ``tests/`` (not ``src/``) — this is our internal optimization
tool, not a shipped feature of the kit. Reuses ``langgraph_kit.evals``
primitives (``LLMJudgeMetric``, ``TraceData``, rule-based metrics)
rather than re-implementing them.

Workflow
--------
1. Author a scenario YAML under ``scenarios/<target>/``.
2. Author a variant prompt under ``variants/<target>/<variant>.md``.
3. Run baseline + variant: ``python -m tests.prompt_bench.run run --target <target> --variant <variant>``.
4. Diff the two reports: ``python -m tests.prompt_bench.run diff --base base.json --variant variant.json``.

See ``tests/prompt_bench/README.md`` for the full iteration loop and
acceptance criteria.
"""

from __future__ import annotations
