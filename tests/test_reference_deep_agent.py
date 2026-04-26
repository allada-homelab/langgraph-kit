"""Tests for reference_deep_agent build function, RuntimeStateMiddleware, and StopHooksMiddleware."""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_kit.core.orchestration.workers import GENERAL_WORKERS
from langgraph_kit.core.resilience.runtime_state import RuntimeStateMiddleware
from langgraph_kit.core.resilience.stop_hooks import (
    StopHooksMiddleware,
    TurnTelemetryStopHook,
)
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


# ---------------------------------------------------------------------------
# deferred_tools auto-gating
# ---------------------------------------------------------------------------
# The ``deferred_tools_awareness`` activation section tells the LLM to
# call ``tool_search`` to discover capabilities that aren't bound to its
# tool surface. If the ``DeferredToolRegistry`` is empty, honoring that
# instruction produces an always-empty search, which on recursion-bound
# runs manifests as spinning on ``tool_search``. The builder must gate
# activation on whether the registry is actually populated.

_DEFERRED_SECTION_MARKER = "use the tool_search tool to discover"


def test_empty_deferred_registry_does_not_activate_deferred_tools_condition(
    mock_store: Any,
) -> None:
    """No ``configure_deferred_tools=`` → no deferred_tools_awareness in prompt."""
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
            agent_name="empty-deferred",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert _DEFERRED_SECTION_MARKER not in system_prompt


def test_populated_deferred_registry_activates_deferred_tools_condition(
    mock_store: Any,
) -> None:
    """``configure_deferred_tools=`` that populates the registry flips the condition on."""
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.graphs._builder import build_deep_agent

    async def _runtime_tool() -> str:
        return "ok"

    def _configure_deferred(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="runtime_tool",
                name="runtime_tool",
                description="a tool the LLM must discover via tool_search",
                fn=_runtime_tool,
                risk=ToolRisk.READ_ONLY,
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
        build_deep_agent(
            agent_name="populated-deferred",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
            configure_deferred_tools=_configure_deferred,
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert _DEFERRED_SECTION_MARKER in system_prompt


def test_explicit_deferred_tools_condition_with_empty_registry_is_stripped(
    mock_store: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit ``conditions={"deferred_tools"}`` + empty registry → stripped with warning.

    Honoring the condition would push the LLM toward an always-empty
    ``tool_search``. Fail loud (warn) and drop the condition so the
    build stays usable.
    """
    from langgraph_kit.graphs._builder import build_deep_agent

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
        caplog.at_level("WARNING", logger="langgraph_kit.graphs._builder"),
    ):
        build_deep_agent(
            agent_name="explicit-empty",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
            conditions={"memory", "deferred_tools", "skills"},
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert _DEFERRED_SECTION_MARKER not in system_prompt

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("'deferred_tools' was requested" in msg for msg in warnings), warnings


def test_empty_deferred_registry_does_not_bind_search_tools_to_llm(
    mock_store: Any,
) -> None:
    """No ``configure_deferred_tools=`` → ``tool_search``/``call_deferred_tool`` must not appear on the LLM tool surface.

    Suppressing the deferred_tools prompt section stops the kit from
    instructing the model to search, but suggestible models (Qwen et al)
    call any tool they can see. An always-empty ``tool_search`` wedges
    recursion-bound runs in a discovery loop the prompt never mentions.
    Gate the tool registration the same way the prompt section is gated.
    """
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
            agent_name="empty-deferred-no-search-tools",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
        )

    compiled_tools = deepagents_mod.create_deep_agent.call_args.kwargs["tools"]
    tool_names = {
        getattr(fn, "__name__", getattr(fn, "name", None)) for fn in compiled_tools
    }
    assert "tool_search" not in tool_names, (
        "tool_search must not reach the LLM when the deferred registry is empty"
    )
    assert "call_deferred_tool" not in tool_names, (
        "call_deferred_tool must not reach the LLM when the deferred registry is empty"
    )


def test_populated_deferred_registry_binds_search_tools_to_llm(
    mock_store: Any,
) -> None:
    """Populated deferred registry → ``tool_search`` + ``call_deferred_tool`` on the LLM surface."""
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.graphs._builder import build_deep_agent

    async def _runtime_tool() -> str:
        return "ok"

    def _configure_deferred(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="runtime_tool",
                name="runtime_tool",
                description="a tool the LLM must discover via tool_search",
                fn=_runtime_tool,
                risk=ToolRisk.READ_ONLY,
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
        build_deep_agent(
            agent_name="populated-deferred-search-tools-present",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
            configure_deferred_tools=_configure_deferred,
        )

    compiled_tools = deepagents_mod.create_deep_agent.call_args.kwargs["tools"]
    tool_names = {
        getattr(fn, "__name__", getattr(fn, "name", None)) for fn in compiled_tools
    }
    assert "tool_search" in tool_names
    assert "call_deferred_tool" in tool_names


def test_explicit_deferred_tools_condition_with_populated_registry_is_kept(
    mock_store: Any,
) -> None:
    """Explicit ``conditions={"deferred_tools"}`` + populated registry → section stays."""
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.graphs._builder import build_deep_agent

    async def _runtime_tool() -> str:
        return "ok"

    def _configure_deferred(registry: Any) -> None:
        registry.register(
            ToolCapability(
                id="runtime_tool",
                name="runtime_tool",
                description="a tool the LLM must discover via tool_search",
                fn=_runtime_tool,
                risk=ToolRisk.READ_ONLY,
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
        build_deep_agent(
            agent_name="explicit-populated",
            core_sections=[],
            subagents=[],
            checkpointer=MagicMock(),
            store=mock_store,
            conditions={"memory", "deferred_tools", "skills"},
            configure_deferred_tools=_configure_deferred,
        )

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert _DEFERRED_SECTION_MARKER in system_prompt


# ---------------------------------------------------------------------------
# TurnTelemetryStopHook tests
# ---------------------------------------------------------------------------
# The hook is the default observability hook wired by
# ``build_reference_deep_agent``. It must be non-blocking and emit a
# stable debug log on every turn so the StopHooksMiddleware path is
# exercised end-to-end without affecting agent behavior.


class TestTurnTelemetryStopHook:
    @pytest.mark.asyncio
    async def test_logs_message_and_tool_call_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook emits one debug line with the expected counts."""
        hook = TurnTelemetryStopHook()
        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="working on it",
                    tool_calls=[
                        {"name": "search", "args": {}, "id": "tc1"},
                        {"name": "read", "args": {}, "id": "tc2"},
                    ],
                ),
            ]
        }
        with caplog.at_level(
            logging.DEBUG, logger="langgraph_kit.core.resilience.stop_hooks"
        ):
            await hook.on_turn_complete(state)

        records = [r for r in caplog.records if r.name.endswith("stop_hooks")]
        assert any(
            "messages=2" in r.getMessage() and "tool_calls=2" in r.getMessage()
            for r in records
        ), [r.getMessage() for r in records]

    @pytest.mark.asyncio
    async def test_zero_tool_calls_when_last_is_human(self) -> None:
        """A trailing HumanMessage produces tool_calls=0 (no AIMessage)."""
        hook = TurnTelemetryStopHook()
        await hook.on_turn_complete(
            {"messages": [AIMessage(content="ok"), HumanMessage(content="next")]}
        )

    @pytest.mark.asyncio
    async def test_handles_missing_messages_key(self) -> None:
        """Empty / missing state keys must not raise."""
        hook = TurnTelemetryStopHook()
        await hook.on_turn_complete({})
        await hook.on_turn_complete({"messages": []})
        await hook.on_turn_complete({"messages": "not-a-list"})

    @pytest.mark.asyncio
    async def test_is_non_blocking(self) -> None:
        """Hook declares blocking=False so StopHooksMiddleware swallows failures."""
        hook = TurnTelemetryStopHook()
        assert hook.blocking is False


# ---------------------------------------------------------------------------
# build_reference_deep_agent: stop_hooks wiring
# ---------------------------------------------------------------------------


def _build_reference_with_capture(
    mock_store: Any,
    *,
    enable_default_stop_hooks: bool | None = None,
    extra_stop_hooks: list[Any] | None = None,
    enable_default_deferred_tools: bool | None = None,
    extra_deferred_tools: Any | None = None,
    enable_default_custom_tools: bool | None = None,
    extra_configure_tools: Any | None = None,
    enable_default_extra_providers: bool | None = None,
    extra_providers: list[Any] | None = None,
) -> MagicMock:
    """Helper: build the reference graph under mocked deepagents+llm.

    Returns the mocked ``deepagents.create_deep_agent`` so callers can
    inspect the kwargs the builder forwarded.
    """
    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    kwargs: dict[str, Any] = {"checkpointer": MagicMock(), "store": mock_store}
    if enable_default_stop_hooks is not None:
        kwargs["enable_default_stop_hooks"] = enable_default_stop_hooks
    if extra_stop_hooks is not None:
        kwargs["extra_stop_hooks"] = extra_stop_hooks
    if enable_default_deferred_tools is not None:
        kwargs["enable_default_deferred_tools"] = enable_default_deferred_tools
    if extra_deferred_tools is not None:
        kwargs["extra_deferred_tools"] = extra_deferred_tools
    if enable_default_custom_tools is not None:
        kwargs["enable_default_custom_tools"] = enable_default_custom_tools
    if extra_configure_tools is not None:
        kwargs["extra_configure_tools"] = extra_configure_tools
    if enable_default_extra_providers is not None:
        kwargs["enable_default_extra_providers"] = enable_default_extra_providers
    if extra_providers is not None:
        kwargs["extra_providers"] = extra_providers
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_deep_agent(**kwargs)
    return deepagents_mod.create_deep_agent


def test_reference_default_stop_hook_is_wired(mock_store: Any) -> None:
    """Default build attaches a TurnTelemetryStopHook via StopHooksMiddleware."""
    create = _build_reference_with_capture(mock_store)

    middleware = create.call_args.kwargs["middleware"]
    stop_mw = next(m for m in middleware if isinstance(m, StopHooksMiddleware))
    hooks = stop_mw._hooks
    assert any(isinstance(h, TurnTelemetryStopHook) for h in hooks), (
        "TurnTelemetryStopHook must be wired by default; otherwise the "
        "reference docstring's 'stop hooks' claim is false"
    )


def test_reference_default_stop_hook_can_be_disabled(mock_store: Any) -> None:
    """``enable_default_stop_hooks=False`` opts out of the telemetry hook."""
    create = _build_reference_with_capture(mock_store, enable_default_stop_hooks=False)

    middleware = create.call_args.kwargs["middleware"]
    stop_mw = next(m for m in middleware if isinstance(m, StopHooksMiddleware))
    assert not any(isinstance(h, TurnTelemetryStopHook) for h in stop_mw._hooks)


def test_reference_extra_stop_hooks_appended_after_default(mock_store: Any) -> None:
    """``extra_stop_hooks=`` runs *after* the default so user hooks observe the same state."""

    class _Marker:
        blocking = False

        async def on_turn_complete(self, state: Any) -> None:
            return None

    extra = _Marker()
    create = _build_reference_with_capture(mock_store, extra_stop_hooks=[extra])

    middleware = create.call_args.kwargs["middleware"]
    stop_mw = next(m for m in middleware if isinstance(m, StopHooksMiddleware))
    hooks = stop_mw._hooks
    # Default hook present, extra appended after it.
    assert any(isinstance(h, TurnTelemetryStopHook) for h in hooks)
    assert hooks[-1] is extra


def test_reference_extra_only_when_default_disabled(mock_store: Any) -> None:
    """Disabling the default leaves only the caller-supplied hooks."""

    class _Marker:
        blocking = False

        async def on_turn_complete(self, state: Any) -> None:
            return None

    extra = _Marker()
    create = _build_reference_with_capture(
        mock_store, enable_default_stop_hooks=False, extra_stop_hooks=[extra]
    )

    middleware = create.call_args.kwargs["middleware"]
    stop_mw = next(m for m in middleware if isinstance(m, StopHooksMiddleware))
    hooks = stop_mw._hooks
    assert hooks == [extra]


# ---------------------------------------------------------------------------
# build_reference_deep_agent: deferred-tool wiring
# ---------------------------------------------------------------------------
# The reference build must populate the DeferredToolRegistry by default
# so the tool_search / call_deferred_tool discovery loop is exercised.
# With the registry empty the builder strips both tools from the active
# surface and the deferred_tools prompt section is gated off — the
# showcase disappears entirely. These tests pin the wiring.


def test_reference_default_deferred_tools_are_populated(mock_store: Any) -> None:
    """Default build registers demo tools; tool_search + call_deferred_tool reach the LLM."""
    create = _build_reference_with_capture(mock_store)

    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "tool_search" in tool_names
    assert "call_deferred_tool" in tool_names
    assert (
        "use the tool_search tool to discover"
        in create.call_args.kwargs["system_prompt"]
    )


def test_reference_default_deferred_tools_can_be_disabled(mock_store: Any) -> None:
    """``enable_default_deferred_tools=False`` opts out — back to empty registry semantics."""
    create = _build_reference_with_capture(
        mock_store, enable_default_deferred_tools=False
    )

    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "tool_search" not in tool_names
    assert "call_deferred_tool" not in tool_names


def test_reference_extra_deferred_tools_runs_after_default(mock_store: Any) -> None:
    """``extra_deferred_tools=`` runs after the default registration so caller IDs win."""
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    captured: dict[str, Any] = {}

    async def _custom_tool() -> str:
        return "custom"

    def _extra(deferred: Any) -> None:
        captured["pre_extra_ids"] = sorted(c.id for c in deferred.list_all())
        deferred.register(
            ToolCapability(
                id="ref_web_fetch_demo",
                name="caller_override",
                description="caller override",
                fn=_custom_tool,
                risk=ToolRisk.READ_ONLY,
            )
        )

    _build_reference_with_capture(mock_store, extra_deferred_tools=_extra)

    assert "ref_web_fetch_demo" in captured["pre_extra_ids"]
    assert "ref_code_indexer_demo" in captured["pre_extra_ids"]
    assert "ref_db_query_demo" in captured["pre_extra_ids"]


def test_reference_extra_deferred_tools_alone_when_default_disabled(
    mock_store: Any,
) -> None:
    """With default disabled, ``extra_deferred_tools=`` is the sole configurator."""
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

    captured: dict[str, Any] = {}

    async def _solo() -> str:
        return ""

    def _extra(deferred: Any) -> None:
        captured["pre_extra_ids"] = sorted(c.id for c in deferred.list_all())
        deferred.register(
            ToolCapability(
                id="solo_tool",
                name="solo_tool",
                description="x",
                fn=_solo,
                risk=ToolRisk.READ_ONLY,
            )
        )

    create = _build_reference_with_capture(
        mock_store,
        enable_default_deferred_tools=False,
        extra_deferred_tools=_extra,
    )

    assert captured["pre_extra_ids"] == []
    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "tool_search" in tool_names


# ---------------------------------------------------------------------------
# build_reference_deep_agent: configure_tools= wiring
# ---------------------------------------------------------------------------
# The reference must showcase the configure_tools= extension point so
# new domain agents have an in-tree pattern other than the
# coding-specific worktree tools. The default registers a single
# read-only ``current_environment`` tool.


def test_reference_default_custom_tool_is_registered(mock_store: Any) -> None:
    """Default build registers ``current_environment`` on the active tool surface."""
    create = _build_reference_with_capture(mock_store)

    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "current_environment" in tool_names


def test_reference_default_custom_tool_can_be_disabled(mock_store: Any) -> None:
    """``enable_default_custom_tools=False`` strips the demo tool from the surface."""
    create = _build_reference_with_capture(
        mock_store, enable_default_custom_tools=False
    )

    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "current_environment" not in tool_names


def test_reference_extra_configure_tools_runs_after_default(mock_store: Any) -> None:
    """``extra_configure_tools=`` runs after the default — caller can inspect/override."""
    captured: dict[str, Any] = {}

    async def custom_caller_tool() -> str:
        return "custom"

    def _extra(registry: Any) -> None:
        captured["pre_extra_caps"] = sorted(c.id for c in registry.list_all())
        # Register a brand-new caller-only tool to confirm the callback ran.
        from langgraph_kit.core.graph_builder.tools import register_tool
        from langgraph_kit.core.tools.capability import ToolRisk

        register_tool(
            registry,
            custom_caller_tool,
            id_prefix="caller",
            tags=["caller"],
            risk=ToolRisk.READ_ONLY,
        )

    create = _build_reference_with_capture(mock_store, extra_configure_tools=_extra)

    # Default ran before extra: the demo tool is present in the registry
    # snapshot taken inside the callback.
    assert "reference_current_environment" in captured["pre_extra_caps"]

    # And the caller's tool reaches the bound surface.
    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "custom_caller_tool" in tool_names


def test_reference_extra_configure_tools_alone_when_default_disabled(
    mock_store: Any,
) -> None:
    """With default disabled, ``extra_configure_tools=`` is the sole tool callback."""
    captured: dict[str, Any] = {}

    async def solo_tool() -> str:
        return ""

    def _extra(registry: Any) -> None:
        captured["pre_extra_caps"] = sorted(c.id for c in registry.list_all())
        from langgraph_kit.core.graph_builder.tools import register_tool
        from langgraph_kit.core.tools.capability import ToolRisk

        register_tool(
            registry,
            solo_tool,
            id_prefix="solo",
            tags=["solo"],
            risk=ToolRisk.READ_ONLY,
        )

    create = _build_reference_with_capture(
        mock_store,
        enable_default_custom_tools=False,
        extra_configure_tools=_extra,
    )

    # Default did NOT run — the demo tool isn't in the snapshot.
    assert "reference_current_environment" not in captured["pre_extra_caps"]
    tools = create.call_args.kwargs["tools"]
    tool_names = {getattr(fn, "__name__", getattr(fn, "name", None)) for fn in tools}
    assert "solo_tool" in tool_names
    assert "current_environment" not in tool_names


# ---------------------------------------------------------------------------
# build_reference_deep_agent: extra_providers wiring
# ---------------------------------------------------------------------------
# The reference must showcase the extra_providers= extension point so
# new domain agents have an exemplar that doesn't assume git (the only
# in-tree provider before this was GitContextProvider on coding_agent).


def _captured_providers(mock_store: Any, **kwargs: Any) -> list[Any]:
    """Build the reference with PromptComposer patched to capture provider list."""
    captured: dict[str, list[Any]] = {}

    from langgraph_kit.core.prompt_assembly.composer import PromptComposer as _Real

    class _SpyComposer(_Real):
        def __init__(self, sections: Any, providers: list[Any]) -> None:
            captured["providers"] = providers
            super().__init__(sections, providers)

    module_patches, _, _ = _mock_deepagents_env()
    full_kwargs: dict[str, Any] = {"checkpointer": MagicMock(), "store": mock_store}
    full_kwargs.update(kwargs)

    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
        patch("langgraph_kit.graphs._builder.PromptComposer", _SpyComposer),
    ):
        build_reference_deep_agent(**full_kwargs)

    return captured["providers"]


def test_reference_default_system_context_provider_is_wired(mock_store: Any) -> None:
    """Default build registers a SystemContextProvider on the prompt composer."""
    from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider

    providers = _captured_providers(mock_store)
    assert any(isinstance(p, SystemContextProvider) for p in providers), (
        f"SystemContextProvider must be wired by default; got {providers!r}"
    )


def test_reference_default_system_context_provider_can_be_disabled(
    mock_store: Any,
) -> None:
    """``enable_default_extra_providers=False`` removes the SystemContextProvider."""
    from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider

    providers = _captured_providers(mock_store, enable_default_extra_providers=False)
    assert not any(isinstance(p, SystemContextProvider) for p in providers)


def test_reference_extra_providers_appended_after_default(mock_store: Any) -> None:
    """Caller-supplied ``extra_providers=`` are appended after the default."""
    from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider

    class _Marker:
        async def provide(self, context: dict[str, Any]) -> str:
            return "marker"

    extra = _Marker()
    providers = _captured_providers(mock_store, extra_providers=[extra])

    # Default present, caller's provider appended after the kit's three +
    # the SystemContextProvider default.
    assert any(isinstance(p, SystemContextProvider) for p in providers)
    assert providers[-1] is extra


def test_reference_extra_providers_alone_when_default_disabled(
    mock_store: Any,
) -> None:
    """With default disabled, only kit defaults + caller's providers appear."""
    from langgraph_kit.core.prompt_assembly.system_context import SystemContextProvider

    class _Marker:
        async def provide(self, context: dict[str, Any]) -> str:
            return "marker"

    extra = _Marker()
    providers = _captured_providers(
        mock_store,
        enable_default_extra_providers=False,
        extra_providers=[extra],
    )

    assert not any(isinstance(p, SystemContextProvider) for p in providers)
    assert providers[-1] is extra


# ---------------------------------------------------------------------------
# build_reference_deep_agent: output_schema= wiring
# ---------------------------------------------------------------------------
# The kwarg flows through to ``build_deep_agent`` which appends
# StructuredOutputMiddleware when set. These tests pin the wiring so the
# kwarg can't be silently dropped.


def test_reference_output_schema_appends_structured_output_middleware(
    mock_store: Any,
) -> None:
    """``output_schema=Schema`` adds a StructuredOutputMiddleware to the stack."""
    from pydantic import BaseModel, Field

    from langgraph_kit.core.resilience.structured_output import (
        StructuredOutputMiddleware,
    )

    class Answer(BaseModel):
        summary: str
        confidence: float = Field(ge=0.0, le=1.0)

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
            output_schema=Answer,
        )

    middleware = deepagents_mod.create_deep_agent.call_args.kwargs["middleware"]
    structured = [m for m in middleware if isinstance(m, StructuredOutputMiddleware)]
    assert structured, (
        "StructuredOutputMiddleware must be in the stack when output_schema= is set"
    )
    assert structured[0]._schema is Answer


def test_reference_no_output_schema_omits_structured_output_middleware(
    mock_store: Any,
) -> None:
    """Default build (no schema) leaves StructuredOutputMiddleware out of the stack."""
    from langgraph_kit.core.resilience.structured_output import (
        StructuredOutputMiddleware,
    )

    create = _build_reference_with_capture(mock_store)
    middleware = create.call_args.kwargs["middleware"]
    assert not any(isinstance(m, StructuredOutputMiddleware) for m in middleware)


# ---------------------------------------------------------------------------
# Coordinator variant
# ---------------------------------------------------------------------------
# ``build_reference_coordinator_agent`` is a thin wrapper that forces
# ``coordinator=True`` on the reference build. The CoordinatorMode
# implementation (read-only filter, prompt sections) lives in
# core/coordinator.py and is unit-tested there; these tests pin the
# wiring at the reference layer.


def test_reference_coordinator_narrows_tools_to_read_only(mock_store: Any) -> None:
    """Coordinator mode strips mutating tools (e.g. save_memory) from the surface."""
    from langgraph_kit.graphs.reference_deep_agent import (
        build_reference_coordinator_agent,
    )

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_coordinator_agent(checkpointer=MagicMock(), store=mock_store)

    tools = deepagents_mod.create_deep_agent.call_args.kwargs["tools"]
    tool_names = {
        str(getattr(fn, "__name__", getattr(fn, "name", "")) or "") for fn in tools
    }
    # save_memory is MUTATING in the standard registration; coordinator
    # mode must filter it out.
    assert "save_memory" not in tool_names, (
        f"save_memory must be stripped in coordinator mode; got {sorted(tool_names)}"
    )


def test_reference_coordinator_includes_delegation_sections(mock_store: Any) -> None:
    """Coordinator system prompt must include the delegation/synthesis sections."""
    from langgraph_kit.graphs.reference_deep_agent import (
        build_reference_coordinator_agent,
    )

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_coordinator_agent(checkpointer=MagicMock(), store=mock_store)

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert "# Coordinator Mode" in system_prompt
    assert "# Delegation Rules" in system_prompt
    assert "# Synthesis Discipline" in system_prompt


def test_reference_coordinator_uses_distinct_agent_name(mock_store: Any) -> None:
    """Coordinator build uses ``reference-coordinator`` so logs distinguish profiles."""
    from langgraph_kit.graphs.reference_deep_agent import (
        build_reference_coordinator_agent,
    )

    module_patches, deepagents_mod, _ = _mock_deepagents_env()
    with (
        patch.dict(sys.modules, module_patches),
        patch(
            "langgraph_kit.graphs._builder.build_llm",
            return_value=MagicMock(name="fake_llm"),
        ),
    ):
        build_reference_coordinator_agent(checkpointer=MagicMock(), store=mock_store)

    assert (
        deepagents_mod.create_deep_agent.call_args.kwargs["name"]
        == "reference-coordinator"
    )


def test_reference_coordinator_rejects_explicit_coordinator_kwarg(
    mock_store: Any,
) -> None:
    """Passing ``coordinator=`` to the wrapper raises — the wrapper forces True."""
    from langgraph_kit.graphs.reference_deep_agent import (
        build_reference_coordinator_agent,
    )

    with pytest.raises(TypeError, match="forces coordinator=True"):
        build_reference_coordinator_agent(
            checkpointer=MagicMock(), store=mock_store, coordinator=False
        )


def test_reference_default_is_not_coordinator(mock_store: Any) -> None:
    """Default ``build_reference_deep_agent`` does NOT activate coordinator mode."""
    create = _build_reference_with_capture(mock_store)
    assert create.call_args.kwargs["name"] == "reference-deep-agent"

    system_prompt = create.call_args.kwargs["system_prompt"]
    assert "# Coordinator Mode" not in system_prompt
