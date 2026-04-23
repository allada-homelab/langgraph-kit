"""Cluster D — plugin-surface edge cases not covered by the happy path.

``test_plugins_e2e.py`` covers the happy path (tool+section contribute
to the graph). This file fills in the edges: section-only plugins,
tool-only plugins, id-collision precedence, multi-plugin composition.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.plugins.registry import PluginContribution, PluginRegistry
from langgraph_kit.core.prompt_assembly.sections import PromptSection, SectionStability
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    capturing_scripted_llm,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


_SECTION_MARKER_A = "__PLUGIN_A_MARKER__"
_SECTION_MARKER_B = "__PLUGIN_B_MARKER__"
_EXTENSION_MARKER = "Plugin-provided extensions may contribute"


async def pingA() -> str:
    """Plugin A's ping tool; returns pong-A."""
    return "pong-A"


async def pingB() -> str:
    """Plugin B's ping tool; returns pong-B."""
    return "pong-B"


@pytest.mark.asyncio
async def test_section_only_plugin_still_activates_extensions(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A plugin that contributes only a section must still auto-activate `extensions`.

    The condition's gate is "does a plugin contribute ANYTHING" — not
    specifically "does a plugin contribute a tool". Section-only
    contributions are legitimate (e.g. context guidance without new
    capabilities) and the prompt should be coherent about their
    presence.
    """
    plugin = PluginContribution(
        plugin_id="section-only",
        sections=[
            PromptSection(
                id="section_only",
                content=f"# Section Only\n{_SECTION_MARKER_A} — no tools contributed.",
                stability=SectionStability.STABLE,
                priority=30,
            )
        ],
    )
    registry = PluginRegistry()
    registry.register(plugin)

    capturing = capturing_scripted_llm([answer("hi")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-section-only",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "plug-section-only"}},  # pyright: ignore[reportArgumentType]
    )

    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _SECTION_MARKER_A in prompt, "Plugin section didn't reach prompt"
    assert _EXTENSION_MARKER in prompt, (
        "Section-only plugin should still flip the extensions condition on. "
        "If this fails, the gate has accidentally been narrowed to 'tools must exist'."
    )


@pytest.mark.asyncio
async def test_tool_only_plugin_activates_extensions(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A plugin with only tools (no sections) still activates extensions."""
    plugin = PluginContribution(
        plugin_id="tool-only",
        tools=[
            ToolCapability(
                id="pingA",
                name="pingA",
                description="Ping from plugin A (tool-only).",
                fn=pingA,
                risk=ToolRisk.READ_ONLY,
            )
        ],
    )
    registry = PluginRegistry()
    registry.register(plugin)

    capturing = capturing_scripted_llm(
        [tool_call_turn("pingA"), answer("done")]
    )
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-tool-only",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="ping A")]},
        config={"configurable": {"thread_id": "plug-tool-only"}},  # pyright: ignore[reportArgumentType]
    )

    assert_tool_invoked(result, "pingA")
    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _EXTENSION_MARKER in prompt, (
        "Tool-only plugin must flip extensions condition on."
    )


@pytest.mark.asyncio
async def test_multiple_plugins_compose_sections_and_tools(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Two plugins: both sections reach the prompt, both tools callable."""
    plugin_a = PluginContribution(
        plugin_id="plugin-a",
        tools=[
            ToolCapability(
                id="pingA",
                name="pingA",
                description="A",
                fn=pingA,
                risk=ToolRisk.READ_ONLY,
            )
        ],
        sections=[
            PromptSection(
                id="plugin_a_section",
                content=f"# A\n{_SECTION_MARKER_A}",
                stability=SectionStability.STABLE,
                priority=30,
            )
        ],
    )
    plugin_b = PluginContribution(
        plugin_id="plugin-b",
        tools=[
            ToolCapability(
                id="pingB",
                name="pingB",
                description="B",
                fn=pingB,
                risk=ToolRisk.READ_ONLY,
            )
        ],
        sections=[
            PromptSection(
                id="plugin_b_section",
                content=f"# B\n{_SECTION_MARKER_B}",
                stability=SectionStability.STABLE,
                priority=31,
            )
        ],
    )
    registry = PluginRegistry()
    registry.register(plugin_a)
    registry.register(plugin_b)

    capturing = capturing_scripted_llm(
        [
            tool_call_turn("pingA"),
            tool_call_turn("pingB"),
            answer("done"),
        ]
    )
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-multi",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="ping both")]},
        config={"configurable": {"thread_id": "plug-multi"}},  # pyright: ignore[reportArgumentType]
    )

    # Both tools executed.
    a_msg = assert_tool_invoked(result, "pingA")
    b_msg = assert_tool_invoked(result, "pingB")
    assert "pong-A" in str(a_msg.content)
    assert "pong-B" in str(b_msg.content)

    # Both section markers landed in the prompt.
    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _SECTION_MARKER_A in prompt and _SECTION_MARKER_B in prompt, (
        f"Multi-plugin section composition missing a marker in the prompt. "
        f"A present: {_SECTION_MARKER_A in prompt}, "
        f"B present: {_SECTION_MARKER_B in prompt}"
    )


@pytest.mark.asyncio
async def test_empty_plugin_registry_does_not_activate_extensions(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """An empty PluginRegistry passed explicitly must NOT auto-activate extensions.

    The gate is "does any plugin contribute something", not "did the
    caller bother to construct a registry". Avoids prompt bloat on
    builds that pass an empty registry for plumbing convenience.
    """
    registry = PluginRegistry()  # Empty — no register() calls.
    # No contributions → no extensions.

    capturing = capturing_scripted_llm([answer("hi")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-empty",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "plug-empty"}},  # pyright: ignore[reportArgumentType]
    )

    prompt = "\n".join(str(getattr(m, "content", "")) for m in capturing.captured_calls[0])
    assert _EXTENSION_MARKER not in prompt, (
        "Empty PluginRegistry must not flip extensions on."
    )
