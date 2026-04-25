"""Orchestration: declarative worker (sub-agent) definitions.

What this shows
---------------
- The shape of a worker definition (``name``, ``description``,
  ``system_prompt``) — same dict the deepagents framework consumes
- The kit's pre-composed lists: ``GENERAL_WORKERS`` (researcher /
  implementer / verifier) and ``CODING_WORKERS`` (specialist verifier
  for code changes)
- How a domain-specific agent extends the catalogue: add a custom
  worker dict, then merge into the base list at build time

No LLM. The ``reference_deep_agent`` and ``coding_agent`` builders
already wire ``GENERAL_WORKERS`` / ``CODING_WORKERS`` into their
``build_deep_agent(subagents=...)`` call — this demo just inspects the
definitions so it's clear what gets shipped.

How to run
----------
    uv run python -m examples.orchestration_workers

Expected output
---------------
    GENERAL_WORKERS: 3 workers
      - researcher  (Deep codebase research and investigation. ...)
      - implementer (Focused code implementation within a bounded scope. ...)
      - verifier    (Independent verification of changes. ...)
    Custom mix: GENERAL_WORKERS + 'devops_runner' = 4 workers
"""

from __future__ import annotations

from typing import Any

from examples._lib import banner, line


def main() -> None:
    banner("orchestration_workers")

    from langgraph_kit.core.orchestration.workers import (
        CODING_WORKERS,
        GENERAL_WORKERS,
    )

    line(f"GENERAL_WORKERS: {len(GENERAL_WORKERS)} workers")
    for worker in GENERAL_WORKERS:
        # Show only the first 60 chars of description so the line stays scannable.
        desc = worker["description"]
        snippet = desc.split(".")[0] + "."
        line(f"  - {worker['name']:<11} ({snippet})")

    line(f"\nCODING_WORKERS: {len(CODING_WORKERS)} workers")
    for worker in CODING_WORKERS:
        line(f"  - {worker['name']}")

    # Build a custom mix: take the general lineup and add a domain
    # worker. Same pattern domain agents use — see
    # src/langgraph_kit/graphs/coding_agent.py for the real-world variant.
    devops_runner: dict[str, Any] = {
        "name": "devops_runner",
        "description": "Runs deployment and infra commands. Use for ops-only steps.",
        "system_prompt": (
            "You are an operations specialist. Run only the requested "
            "command and report its exit code + last 20 lines of output."
        ),
    }
    custom_mix = [*GENERAL_WORKERS, devops_runner]
    line(f"\nCustom mix: GENERAL_WORKERS + 'devops_runner' = {len(custom_mix)} workers")
    for worker in custom_mix:
        line(f"  - {worker['name']}")


if __name__ == "__main__":
    main()
