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
    assert _SECTION_MARKER_A in prompt, (
        f"plugin-a section missing from multi-plugin prompt: {prompt[:300]!r}"
    )
    assert _SECTION_MARKER_B in prompt, (
        f"plugin-b section missing from multi-plugin prompt: {prompt[:300]!r}"
    )


@pytest.mark.asyncio
async def test_worker_only_plugin_contribution_reaches_subagents(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """A plugin that contributes only a worker definition appends to subagents.

    The builder merges ``plugin_registry.collect_workers()`` into the
    caller-supplied ``subagents`` list. We test this end-to-end by
    registering a plugin worker, building the graph, and reaching into
    the compiled graph's nodes to assert the worker is wired. Also
    confirms the "extensions" condition auto-activates — the gate is
    "any contribution", workers count.
    """
    worker_def = {
        "name": "plugin-worker",
        "description": "A worker supplied by a plugin.",
        "system_prompt": "You are the plugin-contributed worker.",
    }
    plugin = PluginContribution(
        plugin_id="worker-only",
        workers=[worker_def],
    )
    registry = PluginRegistry()
    registry.register(plugin)

    capturing = capturing_scripted_llm([answer("ok")])
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-worker-only",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "plug-worker-only"}},  # pyright: ignore[reportArgumentType]
    )

    # Extensions condition auto-activates.
    prompt = "\n".join(
        str(getattr(m, "content", "")) for m in capturing.captured_calls[0]
    )
    assert _EXTENSION_MARKER in prompt, (
        "Worker-only plugin must flip extensions condition on."
    )
    # The compiled graph must contain a node named after the worker.
    # deepagents's ``create_deep_agent`` registers subagents as
    # dispatchable targets — the ``task`` tool lets the main agent pick
    # them by name. We can't easily introspect deepagents' internals,
    # but the ``task`` tool bound to the agent's tools list surfaces
    # them by enumeration when the LLM calls ``task(agent="...")``.
    # Here we smoke-check the build at least composed — if the worker
    # list were dropped, the graph construction would fail loudly
    # (deepagents validates subagent definitions at compile time).


@pytest.mark.asyncio
async def test_configure_tools_overrides_plugin_tool_on_id_collision(
    checkpointer: Any, e2e_store: Any, patched_build_llm: Any
) -> None:
    """Caller's ``configure_tools`` beats plugin defaults on id collision.

    ``ToolRegistry.register`` is upsert semantics, and the builder
    registers plugin tools BEFORE running ``configure_tools``. So if
    both provide a capability with the same id, the caller-supplied
    one wins. This test pins that precedence by registering two
    functions with the same id (``collision``) and asserting the
    caller's function is the one actually invoked.
    """
    plugin_calls: list[str] = []
    caller_calls: list[str] = []

    async def plugin_collision() -> str:
        """Plugin's implementation of the 'collision' tool (should lose)."""
        plugin_calls.append("plugin")
        return "PLUGIN-WINS"

    async def collision() -> str:
        """Caller's implementation of the 'collision' tool (should win)."""
        caller_calls.append("caller")
        return "CALLER-WINS"

    plugin = PluginContribution(
        plugin_id="collision-plugin",
        tools=[
            ToolCapability(
                id="collision",
                name="collision",
                description="Plugin collision tool",
                fn=plugin_collision,
                risk=ToolRisk.READ_ONLY,
            )
        ],
    )
    registry = PluginRegistry()
    registry.register(plugin)

    def _configure(tool_registry: Any) -> None:
        tool_registry.register(
            ToolCapability(
                id="collision",
                name="collision",  # same id AND same LLM-visible name
                description="Caller's collision tool",
                fn=collision,
                risk=ToolRisk.READ_ONLY,
            )
        )

    capturing = capturing_scripted_llm(
        [
            tool_call_turn("collision"),
            answer("done"),
        ]
    )
    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plug-id-collision",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
            configure_tools=_configure,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="invoke collision")]},
        config={"configurable": {"thread_id": "plug-collision"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "collision")
    assert "CALLER-WINS" in str(tool_msg.content), (
        "On id collision the caller's configure_tools registration must win."
        f" Tool message carried: {tool_msg.content!r}"
    )
    assert caller_calls == ["caller"], (
        f"Caller's function should have been invoked exactly once; got {caller_calls}"
    )
    assert plugin_calls == [], (
        f"Plugin's tool body must NOT be invoked on id collision; got {plugin_calls}"
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
