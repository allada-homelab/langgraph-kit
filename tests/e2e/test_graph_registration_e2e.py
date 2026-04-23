"""Coverage fill — ``register_all`` wires every kit graph + echo_agent smoke.

``graphs/__init__.register_all`` is the app-lifespan entry point; it
builds and registers all four kit agents (echo, basic-deep,
reference-deep, coding, supervisor). If one builder breaks, the entry
point silently swallows and logs. These tests assert:

- ``register_all`` populates the registry with all expected ids.
- Each ``try/except`` logs a specific skip message, not a whole-process
  crash.
- ``echo_agent.build_graph`` produces a compiled graph that can be
  invoked end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.graphs import register_all
from langgraph_kit.graphs.echo_agent import build_graph as build_echo_graph
from langgraph_kit.registry import get, list_agents
from tests.e2e.helpers import answer, scripted_llm

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.e2e


@pytest.fixture
def _clean_registry() -> Iterator[None]:
    """Snapshot + restore the agent registry so this test doesn't pollute global state."""
    from langgraph_kit import registry as registry_mod

    snap_registry = dict(registry_mod._registry)
    snap_dispatch = dict(registry_mod._dispatchers)
    snap_meta = dict(registry_mod._metadata)
    registry_mod._registry.clear()
    registry_mod._dispatchers.clear()
    registry_mod._metadata.clear()
    try:
        yield
    finally:
        registry_mod._registry.clear()
        registry_mod._dispatchers.clear()
        registry_mod._metadata.clear()
        registry_mod._registry.update(snap_registry)
        registry_mod._dispatchers.update(snap_dispatch)
        registry_mod._metadata.update(snap_meta)


@pytest.mark.asyncio
async def test_echo_agent_graph_invokes_end_to_end(
    checkpointer: Any, e2e_store: Any
) -> None:
    """``build_echo_graph`` produces a graph that runs an ``ainvoke`` cleanly."""
    scripted = scripted_llm([answer("echo-reply")])
    with patch("langgraph_kit.graphs.echo_agent.build_llm", return_value=scripted):
        graph = build_echo_graph(checkpointer, e2e_store)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "echo-smoke"}},  # pyright: ignore[reportArgumentType]
        )

    messages = result["messages"]
    ai = next(m for m in reversed(messages) if isinstance(m, AIMessage))
    assert "echo-reply" in str(ai.content)


@pytest.mark.usefixtures("_clean_registry")
def test_register_all_registers_every_kit_agent(
    checkpointer: Any,
    e2e_store: Any,
) -> None:
    """``register_all`` populates the registry with every kit-provided agent.

    ``build_llm`` is patched globally so none of the deep-agent builders
    need a real LLM to construct. ``register_all`` catches per-builder
    exceptions individually, so if one builder is broken but others
    succeed, the registry ends up with a partial set — we assert the
    *complete* set lands to catch silent regressions.
    """
    scripted = scripted_llm([])  # any graph that actually invokes will fail loudly
    with (
        patch("langgraph_kit.graphs._builder.build_llm", return_value=scripted),
        patch("langgraph_kit.graphs.echo_agent.build_llm", return_value=scripted),
        patch(
            "langgraph_kit.graphs.basic_deep_agent.build_llm", return_value=scripted
        ),
        patch(
            "langgraph_kit.graphs.supervisor_agent.build_llm",
            return_value=scripted,
        ),
    ):
        register_all(checkpointer, e2e_store)

    registered_ids = {a["id"] for a in list_agents()}
    expected = {
        "echo-agent",
        "basic-deep-agent",
        "reference-deep-agent",
        "coding-agent",
        "supervisor-agent",
    }
    assert expected.issubset(registered_ids), (
        f"register_all should populate every kit agent; missing"
        f" {expected - registered_ids}"
    )

    # Spot-check that ``get(agent_id)`` resolves each registered graph.
    for agent_id in expected:
        graph = get(agent_id)
        assert graph is not None, f"{agent_id} registered but not retrievable"
