"""End-to-end scenarios for the plugin contribution surface.

A ``PluginContribution`` bundles tools + prompt sections + workers that
a downstream extension wants merged into the agent. Unit tests confirm
each piece lands in its respective registry, but only an e2e run can
prove the contributions reach the LLM (section in the system prompt)
and actually execute (tool callable in the compiled graph).
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


async def ping() -> str:
    """Plugin-contributed tool body. Returns a distinctive marker the
    assertion can match on so no other tool/prompt text can spoof it.
    """
    return "pong-from-plugin"


_PLUGIN_SECTION_MARKER = "__TESTPLUGIN_SECTION_MARKER__"
_EXTENSION_AWARENESS_MARKER = "Plugin-provided extensions may contribute"


@pytest.mark.asyncio
async def test_plugin_tool_and_section_reach_running_graph(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Plugin-contributed tool is callable + plugin section reaches the LLM prompt.

    Builds a ``PluginContribution`` with:
    - one tool (``ping() -> "pong-from-plugin"``)
    - one section carrying a distinctive marker string

    Scripts the LLM to call ``ping``, then return a final answer.
    Uses ``capturing_scripted_llm`` so the test can inspect the exact
    system prompt the LLM received.

    Asserts:
    1. The plugin's section marker appears in the prompt (contribution
       reached the SectionRegistry → PromptComposer → LLM).
    2. The ``extensions`` condition auto-activated because a plugin is
       present (its awareness section appears in the prompt).
    3. The scripted ``ping`` call actually reached the plugin's tool
       function and returned "pong-from-plugin" (contribution reached
       the ToolRegistry → bound tool list → dispatch).
    """
    plugin_tool = ToolCapability(
        id="ping",
        name="ping",
        description="Return pong — plugin-contributed tool for the e2e test.",
        fn=ping,
        risk=ToolRisk.READ_ONLY,
    )
    plugin_section = PromptSection(
        id="testplugin_section",
        content=(
            "# Test Plugin\n"
            f"{_PLUGIN_SECTION_MARKER} — plugin-contributed section content."
        ),
        stability=SectionStability.STABLE,
        priority=30,
    )
    plugin = PluginContribution(
        plugin_id="test-plugin",
        tools=[plugin_tool],
        sections=[plugin_section],
    )
    registry = PluginRegistry()
    registry.register(plugin)

    capturing = capturing_scripted_llm(
        [
            tool_call_turn("ping"),
            answer("ping returned: pong-from-plugin"),
        ]
    )

    with patched_build_llm(capturing):
        graph, _ = build_deep_agent(
            agent_name="plugins-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            plugins=registry,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="ping the plugin")]},
        config={"configurable": {"thread_id": "plugins-1"}},  # pyright: ignore[reportArgumentType]
    )

    # 1. Plugin section reached the prompt the LLM actually saw.
    assert capturing.captured_calls, "scripted model was never invoked"
    first_call = capturing.captured_calls[0]
    combined_prompt = "\n".join(str(getattr(m, "content", "")) for m in first_call)
    assert _PLUGIN_SECTION_MARKER in combined_prompt, (
        "Plugin section did not reach the system prompt the LLM received. "
        f"Prompt excerpt: {combined_prompt[:800]!r}"
    )

    # 2. The ``extensions`` condition auto-activated because a plugin is
    # present, so extension_awareness lands in the prompt.
    assert _EXTENSION_AWARENESS_MARKER in combined_prompt, (
        "Plugin was registered but the 'extensions' condition did not "
        "auto-activate — extension_awareness section missing from prompt."
    )

    # 3. Plugin tool actually ran and its output reached the state as a
    # ToolMessage. This exercises the full path: PluginRegistry →
    # build_deep_agent merge → ToolRegistry → bound tools → dispatch.
    ping_msg = assert_tool_invoked(result, "ping")
    assert "pong-from-plugin" in str(ping_msg.content), (
        f"Plugin tool ran but returned unexpected content: {ping_msg.content!r}"
    )
