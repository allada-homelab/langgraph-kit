"""Tests for reference_deep_agent build function, RuntimeStateMiddleware, and StopHooksMiddleware."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
from langgraph_kit.core.resilience.runtime_state import RuntimeStateMiddleware
from langgraph_kit.core.resilience.stop_hooks import StopHooksMiddleware
from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent

# ---------------------------------------------------------------------------
# GENERAL_WORKERS tests
# ---------------------------------------------------------------------------


def test_worker_definitions_valid() -> None:
    """GENERAL_WORKERS has 3 entries with name, description, system_prompt."""
    assert len(GENERAL_WORKERS) == 3
    for defn in GENERAL_WORKERS:
        assert "name" in defn
        assert "description" in defn
        assert "system_prompt" in defn
        assert isinstance(defn["name"], str)
        assert defn["name"]
        assert isinstance(defn["description"], str)
        assert defn["description"]
        assert isinstance(defn["system_prompt"], str)
        assert defn["system_prompt"]


# ---------------------------------------------------------------------------
# RuntimeStateMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_state_middleware_tracks_state() -> None:
    """Create RuntimeStateMiddleware, call abefore_agent, verify state='started'."""
    mw = RuntimeStateMiddleware()
    assert mw.state == "idle"

    await mw.abefore_agent({}, MagicMock())
    assert mw.state == "started"
    assert mw.turn_count == 1


@pytest.mark.asyncio
async def test_runtime_state_middleware_model_call_success() -> None:
    """Mock handler, verify state transitions to 'completed'."""
    mw = RuntimeStateMiddleware()
    await mw.abefore_agent({}, MagicMock())

    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = AsyncMock(return_value=mock_response)
    result = await mw.awrap_model_call(mock_request, mock_handler)

    assert result is mock_response
    assert mw.state == "completed"
    assert mw.stop_reason == "final_answer"


@pytest.mark.asyncio
async def test_runtime_state_middleware_model_call_failure() -> None:
    """Mock handler raises, verify state='failed'."""
    mw = RuntimeStateMiddleware()
    await mw.abefore_agent({}, MagicMock())

    mock_request = MagicMock()
    mock_handler = AsyncMock(side_effect=ValueError("something broke"))

    with pytest.raises(ValueError, match="something broke"):
        await mw.awrap_model_call(mock_request, mock_handler)

    assert mw.state == "failed"
    assert mw.stop_reason is not None
    assert "ValueError" in mw.stop_reason


# ---------------------------------------------------------------------------
# StopHooksMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_hooks_middleware_runs_hooks() -> None:
    """Register a mock hook with on_turn_complete, verify it's called."""
    hook = AsyncMock()
    hook.on_turn_complete = AsyncMock()
    hook.blocking = False

    mw = StopHooksMiddleware(hooks=[hook])

    await mw.aafter_agent({"messages": []}, MagicMock())

    hook.on_turn_complete.assert_awaited_once_with({"messages": []})


@pytest.mark.asyncio
async def test_stop_hooks_middleware_non_blocking_failure() -> None:
    """Hook raises, but middleware doesn't crash (non-blocking)."""
    hook = MagicMock()
    hook.on_turn_complete = AsyncMock(side_effect=RuntimeError("hook failed"))
    hook.blocking = False

    mw = StopHooksMiddleware(hooks=[hook])

    # Should not raise
    await mw.aafter_agent({"messages": []}, MagicMock())

    hook.on_turn_complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# build_reference_deep_agent smoke test
# ---------------------------------------------------------------------------


def test_build_reference_deep_agent_returns_graph(mock_store: Any) -> None:
    """Call build_reference_deep_agent with mock store + mock checkpointer."""
    checkpointer = MagicMock()

    fake_graph = MagicMock(name="compiled_graph")
    # `with_config` returns itself so the identity check below continues to work;
    # real `CompiledStateGraph.with_config` returns a new graph with merged config.
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph
    fake_llm = MagicMock(name="fake_llm")

    # Mock deepagents and its backend submodules so lazy imports resolve
    backends_mod = MagicMock()
    module_patches = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }

    with (
        patch.dict(sys.modules, module_patches),
        patch("langgraph_kit.graphs._builder.build_llm", return_value=fake_llm),
    ):
        graph, _dispatcher = build_reference_deep_agent(
            checkpointer=checkpointer, store=mock_store
        )

    assert graph is fake_graph
    deepagents_mod.create_deep_agent.assert_called_once()
    fake_graph.with_config.assert_called_once_with({"recursion_limit": 100})


def test_build_reference_deep_agent_accepts_recursion_limit_override(
    mock_store: Any,
) -> None:
    """Custom recursion_limit is forwarded to the compiled graph."""
    checkpointer = MagicMock()

    fake_graph = MagicMock(name="compiled_graph")
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph
    fake_llm = MagicMock(name="fake_llm")

    backends_mod = MagicMock()
    module_patches = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }

    with (
        patch.dict(sys.modules, module_patches),
        patch("langgraph_kit.graphs._builder.build_llm", return_value=fake_llm),
    ):
        build_reference_deep_agent(
            checkpointer=checkpointer,
            store=mock_store,
            recursion_limit=500,
        )

    fake_graph.with_config.assert_called_once_with({"recursion_limit": 500})


# ---------------------------------------------------------------------------
# PluginRegistry wiring tests
# ---------------------------------------------------------------------------


def _mock_deepagents_env() -> tuple[dict[str, MagicMock], MagicMock, MagicMock]:
    """Return (module_patches, deepagents_mod, fake_graph) for patching sys.modules."""
    fake_graph = MagicMock(name="compiled_graph")
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph

    backends_mod = MagicMock()
    module_patches: dict[str, MagicMock] = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }
    return module_patches, deepagents_mod, fake_graph


def test_plugin_tools_are_registered_on_active_tool_surface(mock_store: Any) -> None:
    """Plugin-contributed tools must reach the agent's bound tool list.

    Regression test for the pre-existing gap: ``PluginRegistry`` was
    defined but no builder ever called ``collect_tools()`` on it, so
    plugin tools sat in the registry unused. Now ``build_deep_agent``
    pulls from it and the tools show up in ``create_deep_agent(tools=…)``.
    """
    from langgraph_kit.core.plugins.registry import (
        PluginContribution,
        PluginRegistry,
    )
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    async def my_plugin_tool() -> str:
        return "from plugin"

    cap = ToolCapability(
        id="plug_my_tool",
        name="my_plugin_tool",
        description="A tool contributed by a plugin",
        fn=my_plugin_tool,
        risk=ToolRisk.READ_ONLY,
    )
    registry = PluginRegistry()
    registry.register(PluginContribution("test-plugin", tools=[cap]))

    checkpointer = MagicMock()
    module_patches, deepagents_mod, _ = _mock_deepagents_env()

    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=checkpointer, store=mock_store, plugins=registry
        )

    # The tools list passed to create_deep_agent must include our plugin
    # tool. The list is compiled from ToolRegistry.compile_tools() which
    # emits the raw callables; check by identity.
    call_kwargs = deepagents_mod.create_deep_agent.call_args.kwargs
    tools = call_kwargs["tools"]
    assert my_plugin_tool in tools, (
        "Plugin tool must appear in the compiled tool list passed to "
        "create_deep_agent; otherwise the LLM cannot call it"
    )


def test_plugin_sections_reach_the_system_prompt(mock_store: Any) -> None:
    """Plugin-contributed sections must end up in the composed system prompt.

    Same wiring-gap class as plugin tools: sections used to get thrown
    on the floor because ``build_deep_agent`` never touched the plugin
    registry. Now they're merged into the ``SectionRegistry`` before
    prompt composition.
    """
    from langgraph_kit.core.plugins.registry import (
        PluginContribution,
        PluginRegistry,
    )
    from langgraph_kit.core.prompt_assembly.sections import (
        PromptSection,
        SectionStability,
    )

    unique_marker = "PLUGIN_PROMPT_MARKER_ZZZZ"
    plugin_section = PromptSection(
        id="plugin_custom",
        content=f"# Custom plugin rule\n{unique_marker}",
        stability=SectionStability.STABLE,
        priority=80,
    )
    registry = PluginRegistry()
    registry.register(PluginContribution("test-plugin", sections=[plugin_section]))

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=MagicMock(), store=mock_store, plugins=registry
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert unique_marker in system_prompt, (
        "Plugin section content must appear in the composed system prompt"
    )


def test_plugin_workers_are_appended_to_subagents(mock_store: Any) -> None:
    """Plugin-contributed workers must reach the subagents list.

    Appended (not prepended) so caller-declared workers keep priority
    in any deepagents routing that walks the list in order.
    """
    from langgraph_kit.core.plugins.registry import (
        PluginContribution,
        PluginRegistry,
    )

    plugin_worker = {
        "name": "plugin-worker",
        "description": "Contributed by a plugin",
        "system_prompt": "You are a plugin-provided worker.",
    }
    registry = PluginRegistry()
    registry.register(PluginContribution("test-plugin", workers=[plugin_worker]))

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=MagicMock(), store=mock_store, plugins=registry
        )

    merged_subagents = deepagents_mod.create_deep_agent.call_args.kwargs["subagents"]
    # The built-in workers come first, plugin worker last.
    worker_names = [w["name"] for w in merged_subagents]
    assert "plugin-worker" in worker_names
    assert worker_names[-1] == "plugin-worker", (
        "Plugin workers must be APPENDED so caller-declared workers keep list priority"
    )


def test_plugins_accepts_bare_contribution_list(mock_store: Any) -> None:
    """``plugins=`` accepts a list of PluginContribution as an ergonomic shorthand.

    Users who only need a quick inline extension shouldn't have to
    instantiate and populate a PluginRegistry first.
    """
    from langgraph_kit.core.plugins.registry import PluginContribution
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    async def listy() -> str:
        return ""

    cap = ToolCapability(
        id="plug_listy",
        name="listy_plugin_tool",
        description="x",
        fn=listy,
        risk=ToolRisk.READ_ONLY,
    )

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=MagicMock(),
            store=mock_store,
            plugins=[PluginContribution("inline-plugin", tools=[cap])],
        )

    assert listy in deepagents_mod.create_deep_agent.call_args.kwargs["tools"]


def test_plugin_presence_activates_extensions_condition(mock_store: Any) -> None:
    """A non-empty plugin registry auto-adds ``"extensions"`` to conditions.

    The ``extension_awareness`` activation section tells the model that
    plugin capabilities are first-class. Without the auto-activation it
    stays dormant on vanilla builds; with plugins present it must flip on.
    """
    from langgraph_kit.core.plugins.registry import (
        PluginContribution,
        PluginRegistry,
    )
    from langgraph_kit.core.prompt_assembly.sections import (
        PromptSection,
        SectionStability,
    )

    registry = PluginRegistry()
    registry.register(
        PluginContribution(
            "test-plugin",
            sections=[
                PromptSection(
                    id="noise",
                    content="noise",
                    stability=SectionStability.STABLE,
                    priority=10,
                )
            ],
        )
    )

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=MagicMock(), store=mock_store, plugins=registry
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    # The extension_awareness section lives in ACTIVATION_SECTIONS under
    # the "extensions" condition. If that condition is active the prompt
    # contains the section's distinctive lead; otherwise it doesn't.
    assert "Plugin-provided extensions may contribute" in system_prompt


def test_empty_plugin_registry_does_not_activate_extensions(mock_store: Any) -> None:
    """A plugins arg with no contributions must NOT flip the extensions condition.

    The gate is "does a plugin actually contribute something" — not "did
    the caller bother to construct a PluginRegistry". Avoids bloating
    the prompt on builds that pass an empty registry for plumbing
    convenience.
    """
    from langgraph_kit.core.plugins.registry import PluginRegistry

    empty_registry = PluginRegistry()

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(
            checkpointer=MagicMock(), store=mock_store, plugins=empty_registry
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert "Plugin-provided extensions may contribute" not in system_prompt


def test_configure_tools_wins_over_plugin_tool_id_collision(mock_store: Any) -> None:
    """Caller-supplied ``configure_tools`` must override plugin tools on id clash.

    Precedence: plugin tools land first, then the callback runs — so a
    plugin default can ship but a specific consumer can swap it out for
    something project-specific by registering under the same id.
    """
    from langgraph_kit.core.plugins.registry import (
        PluginContribution,
        PluginRegistry,
    )
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    async def plugin_version() -> str:
        return "from plugin"

    async def caller_version() -> str:
        return "from caller override"

    plugin_cap = ToolCapability(
        id="shared_id",
        name="shared_tool",
        description="plugin default",
        fn=plugin_version,
        risk=ToolRisk.READ_ONLY,
    )
    override_cap = ToolCapability(
        id="shared_id",  # SAME id
        name="shared_tool",
        description="caller override",
        fn=caller_version,
        risk=ToolRisk.READ_ONLY,
    )
    registry = PluginRegistry()
    registry.register(PluginContribution("p", tools=[plugin_cap]))

    def _configure_tools(tool_registry: Any) -> None:
        tool_registry.register(override_cap)

    from langgraph_kit.graphs._builder import build_deep_agent

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_deep_agent(
            agent_name="override-test",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
            plugins=registry,
            configure_tools=_configure_tools,
        )

    tools = deepagents_mod.create_deep_agent.call_args.kwargs["tools"]
    assert caller_version in tools
    assert plugin_version not in tools, (
        "configure_tools must run AFTER plugin merge so caller overrides win"
    )
