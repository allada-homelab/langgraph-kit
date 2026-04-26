"""Tests for ``AgentConfig.prompt_overrides`` lifecycle wiring (#43 v2)."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

from langgraph_kit._config import AgentConfig, configure, get_config
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.graphs._builder import build_deep_agent


def _mock_deepagents_env() -> tuple[dict[str, MagicMock], MagicMock, MagicMock]:
    fake_graph = MagicMock(name="compiled_graph")
    fake_graph.with_config.return_value = fake_graph
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph

    backends_mod = MagicMock()
    return (
        {
            "deepagents": deepagents_mod,
            "deepagents.backends": backends_mod,
            "deepagents.backends.composite": backends_mod.composite,
            "deepagents.backends.state": backends_mod.state,
            "deepagents.backends.store": backends_mod.store,
        },
        deepagents_mod,
        fake_graph,
    )


def test_prompt_override_replaces_shipped_section(mock_store: Any) -> None:
    """An override under a shipped section's id replaces it in the composed prompt.

    Verifies the lifecycle path: ``AgentConfig.prompt_overrides`` is
    consulted by ``build_deep_agent`` after every other registration
    so the override wins.
    """
    sentinel_marker = "OVERRIDE_SENTINEL_MARKER_KZKZKZ"
    override = PromptSection(
        id="core_identity",
        version="custom-1",
        content=f"You are a custom override agent. {sentinel_marker}",
        stability=SectionStability.STABLE,
        priority=100,
    )

    original_cfg = get_config()
    try:
        configure(AgentConfig(prompt_overrides={"core_identity": override}))

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
                core_sections=[
                    PromptSection(
                        id="core_identity",
                        content="You are a default agent.",
                        stability=SectionStability.STABLE,
                        priority=100,
                    )
                ],
                subagents=[],
                checkpointer=MagicMock(),
                store=mock_store,
            )
    finally:
        configure(original_cfg)

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert sentinel_marker in system_prompt, system_prompt[:400]
    assert "You are a default agent." not in system_prompt


def test_no_overrides_leaves_shipped_sections_intact(mock_store: Any) -> None:
    """Without any overrides, the shipped section content reaches the prompt unchanged."""
    original_cfg = get_config()
    try:
        configure(AgentConfig())  # default — empty prompt_overrides

        module_patches, deepagents_mod, _ = _mock_deepagents_env()
        with (
            patch.dict(sys.modules, module_patches),
            patch(
                "langgraph_kit.graphs._builder.build_llm",
                return_value=MagicMock(name="fake_llm"),
            ),
        ):
            build_deep_agent(
                agent_name="no-override",
                core_sections=[
                    PromptSection(
                        id="core_identity",
                        content="DEFAULT_CONTENT_AAAAA",
                        stability=SectionStability.STABLE,
                        priority=100,
                    )
                ],
                subagents=[],
                checkpointer=MagicMock(),
                store=mock_store,
            )
    finally:
        configure(original_cfg)

    system_prompt = deepagents_mod.create_deep_agent.call_args.kwargs["system_prompt"]
    assert "DEFAULT_CONTENT_AAAAA" in system_prompt


def test_prompt_overrides_field_is_independent_per_config_instance() -> None:
    """Two AgentConfig instances must not share the same prompt_overrides dict.

    ``field(default_factory=dict)`` per-instance — regression guard
    against accidentally using a shared mutable default.
    """
    a = AgentConfig()
    b = AgentConfig()
    assert a.prompt_overrides is not b.prompt_overrides
    a.prompt_overrides["x"] = PromptSection(
        id="x", content="x", stability=SectionStability.STABLE, priority=1
    )
    assert "x" not in b.prompt_overrides
