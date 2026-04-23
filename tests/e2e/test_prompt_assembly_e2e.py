"""Cluster C — prompt composer edge cases and provider-failure isolation.

Unit tests at ``test_prompt_assembly`` exercise the composer directly
with fake sections/providers. This file pulls the same pieces through
a real graph build so any regression in section merging or context
provider invocation surfaces at composition time (not at some later
point when a downstream feature silently breaks).

Two invariants pinned here:

- **Priority/stability ordering.** Sections are sorted STABLE first,
  then CONDITIONAL, then VOLATILE; within each tier priority
  descending. The composer returns a single string, so the invariant
  is visible as the substring ordering inside the system prompt the
  LLM receives.
- **Provider failure isolation.** A provider that raises must not
  crash prompt assembly — the composer should swallow or short-circuit
  that provider and emit the prompt with the surviving parts. (Without
  this guarantee, one broken plugin provider would brick the agent.)
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, capturing_scripted_llm

pytestmark = pytest.mark.e2e


_HIGH_PRIORITY = PromptSection(
    id="core-high-priority",
    content="HIGH-PRIORITY-SECTION-MARKER-8a2f",
    stability=SectionStability.STABLE,
    priority=100,
)
_LOW_PRIORITY = PromptSection(
    id="core-low-priority",
    content="LOW-PRIORITY-SECTION-MARKER-d71c",
    stability=SectionStability.STABLE,
    priority=10,
)
_VOLATILE_LATE = PromptSection(
    id="volatile-late",
    content="VOLATILE-LATE-MARKER-5e8b",
    stability=SectionStability.VOLATILE,
    priority=999,  # high priority within its tier, but VOLATILE ranks below STABLE
)


@pytest.mark.asyncio
async def test_prompt_sections_ordered_stable_first_then_by_priority(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Composed prompt places STABLE sections before VOLATILE; ties break by priority desc."""
    capturing = capturing_scripted_llm([answer("ok")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="compose-order",
            core_sections=[_HIGH_PRIORITY, _LOW_PRIORITY, _VOLATILE_LATE],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="compose")]},
        config={"configurable": {"thread_id": "compose-order"}},  # pyright: ignore[reportArgumentType]
    )

    assert capturing.captured_calls, "scripted LLM was never invoked"
    prompt = "\n".join(
        str(getattr(m, "content", "")) for m in capturing.captured_calls[0]
    )

    # All three markers must appear.
    for marker in (
        "HIGH-PRIORITY-SECTION-MARKER-8a2f",
        "LOW-PRIORITY-SECTION-MARKER-d71c",
        "VOLATILE-LATE-MARKER-5e8b",
    ):
        assert marker in prompt, (
            f"Section with marker {marker!r} missing from composed prompt"
        )

    high_idx = prompt.index("HIGH-PRIORITY-SECTION-MARKER-8a2f")
    low_idx = prompt.index("LOW-PRIORITY-SECTION-MARKER-d71c")
    volatile_idx = prompt.index("VOLATILE-LATE-MARKER-5e8b")

    # STABLE-tier sort: high priority first.
    assert high_idx < low_idx, (
        f"Higher-priority STABLE section should precede lower-priority STABLE"
        f" section. high_idx={high_idx}, low_idx={low_idx}"
    )
    # STABLE tier precedes VOLATILE tier regardless of in-tier priority.
    assert low_idx < volatile_idx, (
        f"STABLE section (even lower-priority) should precede VOLATILE"
        f" section (even higher-priority). low_idx={low_idx},"
        f" volatile_idx={volatile_idx}"
    )


class _RaisingContextProvider:
    """Provider that always raises — exists to test failure isolation."""

    async def provide(self, context: dict[str, Any]) -> str:
        msg = "intentional provider failure"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_graph_invokes_even_when_extra_context_provider_raises(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """A broken ``extra_providers`` entry must not kill graph construction / invocation.

    The kit composes sections synchronously at build time via
    ``compose_sections_only`` (which does NOT invoke providers), and
    leaves provider execution to later runtime points. The contract
    this test pins: a provider that raises should not crash the agent.

    If this test fails, a broken plugin-contributed provider would
    brick every kit-hosted agent. Right now the behavior is actually
    "providers are loaded for future runtime use but not invoked
    during synchronous prompt composition" — which is itself the
    isolation guarantee. If a future change starts calling providers
    synchronously during build, this test must be extended to verify
    the call site swallows provider exceptions.
    """
    scripted = capturing_scripted_llm([answer("survived")])
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="provider-fail",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            extra_providers=[_RaisingContextProvider()],
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "provider-fail"}},  # pyright: ignore[reportArgumentType]
    )

    assert scripted.captured_calls, (
        "Graph must still reach the LLM despite the failing provider"
    )
    final_contents = [
        str(getattr(m, "content", ""))
        for m in result["messages"]
        if getattr(m, "type", None) == "ai"
    ]
    assert any("survived" in c for c in final_contents), (
        "Failing provider should not block agent completion"
    )
