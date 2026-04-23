"""Cross-cluster meta-invariant tests.

Each test here generalizes a class of bug the e2e layer was built to
catch. The deferred_tools regression was the first instance: a
condition in the prompt instructed the LLM to use a capability that
wasn't actually wired. These invariants ensure that pattern can't
silently recur for any other condition/capability pair.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.prompt_assembly.activation import ACTIVATION_SECTIONS
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import answer, capturing_scripted_llm

pytestmark = pytest.mark.e2e


# Each section's "first meaningful phrase" — enough to detect the
# section in a composed prompt without being overly brittle to minor
# rewordings. If a section's content changes substantially, update here.
_SECTION_MARKERS: dict[str, str] = {
    "deferred_tools_awareness": "use the tool_search tool to discover",
    "skill_activation": "call `list_skills()` to see available skills",
    "extension_awareness": "Plugin-provided extensions may contribute",
    "async_tasks_awareness": "launch long-running tasks in the background",
}


def test_every_activation_section_has_a_known_marker() -> None:
    """Sanity guard for the invariant tests below.

    If ACTIVATION_SECTIONS gains a new entry, _SECTION_MARKERS must be
    updated so the invariant tests can detect presence/absence of that
    section. This test fails the suite rather than letting the new
    section silently skip invariant checks.
    """
    section_ids = {s.id for s in ACTIVATION_SECTIONS}
    known = set(_SECTION_MARKERS.keys())
    missing = section_ids - known
    assert not missing, (
        f"ACTIVATION_SECTIONS gained new entries {missing} — update "
        "_SECTION_MARKERS and add an invariant pair for each."
    )


@pytest.mark.asyncio
async def test_default_build_does_not_activate_deferred_tools_condition(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Default build = empty DeferredToolRegistry ⇒ no deferred_tools_awareness.

    Generalizes the deferred_tools regression guard: the prompt section
    must not fire when the backing capability is empty. Already
    verified in test_deferred_tools_e2e.py; kept here for invariant
    completeness alongside the other conditions.
    """
    capturing = capturing_scripted_llm([answer("hi")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="inv-deferred",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "inv-deferred"}},  # pyright: ignore[reportArgumentType]
    )
    assert capturing.captured_calls
    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _SECTION_MARKERS["deferred_tools_awareness"] not in prompt, (
        "deferred_tools_awareness leaked into a default-build prompt. "
        "Empty DeferredToolRegistry must NOT auto-activate the condition."
    )


@pytest.mark.asyncio
async def test_default_build_does_not_activate_extensions_condition(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Default build = no plugins ⇒ no extension_awareness section in prompt.

    The ``extensions`` condition auto-activates only when a plugin
    actually contributes something (tool, section, or worker). A
    default build with no plugins must NOT advertise plugin-extensions
    — the section would tell the LLM "plugin-provided extensions may
    contribute tools" when none exist.
    """
    capturing = capturing_scripted_llm([answer("hi")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="inv-extensions",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "inv-extensions"}},  # pyright: ignore[reportArgumentType]
    )
    assert capturing.captured_calls
    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _SECTION_MARKERS["extension_awareness"] not in prompt, (
        "extension_awareness leaked into a prompt with no plugins registered. "
        "The condition must auto-gate on a non-empty PluginRegistry."
    )


@pytest.mark.asyncio
async def test_explicit_deferred_tools_condition_with_empty_registry_still_scrubbed(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Even if the caller explicitly asks for ``deferred_tools``, an empty
    registry strips it (with a warning) so the prompt section doesn't reach the LLM.

    This is the "fail loud AND fix automatically" part of the
    deferred_tools gating — the unit test already asserts the warning
    logs; here we just confirm the prompt the LLM receives is scrubbed.
    """
    capturing = capturing_scripted_llm([answer("hi")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="inv-explicit-deferred",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            conditions={"memory", "deferred_tools", "skills", "async_tasks"},
        )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "inv-explicit"}},  # pyright: ignore[reportArgumentType]
    )
    assert capturing.captured_calls
    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _SECTION_MARKERS["deferred_tools_awareness"] not in prompt, (
        "The kit must strip an explicitly-requested deferred_tools condition "
        "when the registry is empty. Prompt excerpt: " + prompt[:300]
    )
