"""Cluster J — A2A (Agent-to-Agent) contrib round-trip.

``invoke_agent_a2a`` wraps a registered graph in A2A Task format —
an ``id``, ``contextId``, ``status``, and ``artifacts`` payload. This
is how external agents discover and talk to kit-hosted agents.

These tests register a minimal kit graph, drive it through
``invoke_agent_a2a``, and assert the returned Task envelope has the
expected shape. Also covers the aggregated Agent Card at
``/.well-known/agent.json``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from langgraph_kit.contrib.a2a import (
    build_agent_card,
    build_aggregated_card,
    invoke_agent_a2a,
)
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.registry import AgentMetadata, register
from tests.e2e.helpers import answer, scripted_llm

pytestmark = pytest.mark.e2e


@pytest.fixture
def registered_echo_agent(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> Iterator[str]:
    """Build and register a minimal kit agent for the test.

    Cleans up the registry slot on teardown so test isolation holds.
    """
    agent_id = "a2a-echo-test"
    scripted = scripted_llm([answer("echo-from-a2a")])
    with patched_build_llm(scripted):
        graph, dispatcher = build_deep_agent(
            agent_name=agent_id,
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    register(
        agent_id,
        graph,
        command_dispatcher=dispatcher,
        metadata=AgentMetadata(
            description="test echo agent for A2A",
            tags=["test", "echo"],
            capabilities=["streaming"],
        ),
    )

    try:
        yield agent_id
    finally:
        # Clean up — the registry is module-level state and we don't
        # want this agent leaking into other tests. There's no public
        # deregister API, so poke the internals directly.
        from langgraph_kit import registry as registry_mod

        registry_mod._registry.pop(agent_id, None)
        registry_mod._dispatchers.pop(agent_id, None)
        registry_mod._metadata.pop(agent_id, None)


@pytest.mark.asyncio
async def test_invoke_agent_a2a_returns_task_envelope(
    registered_echo_agent: str,
) -> None:
    """Happy path: invoke returns a Task with status=completed and text artifact."""
    result = await invoke_agent_a2a(
        registered_echo_agent,
        "hello a2a",
        thread_id="a2a-task-1",
    )

    assert result["status"]["state"] == "completed", (
        f"Expected completed status; got {result['status']!r}"
    )
    assert result["contextId"] == "a2a-task-1"
    assert "id" in result
    assert isinstance(result["id"], str)

    artifacts = result.get("artifacts", [])
    assert artifacts, f"A2A response should include artifacts; got {result}"
    parts = artifacts[0].get("parts", [])
    assert any(
        p.get("kind") == "text" and "echo-from-a2a" in p.get("text", "")
        for p in parts
    ), f"Agent response text missing from Task artifacts: {artifacts!r}"


def test_build_agent_card_shape(registered_echo_agent: str) -> None:
    """build_agent_card returns an A2A-compliant card shape."""
    card = build_agent_card(registered_echo_agent, "https://example.test")
    assert card["url"] == f"https://example.test/a2a/{registered_echo_agent}"
    assert card["description"] == "test echo agent for A2A"
    assert card["capabilities"]["streaming"] is True, (
        "Registered agent advertised streaming capability — card should echo it"
    )
    assert card["skills"], "Card should have at least one skill entry"
    assert card["skills"][0]["id"] == registered_echo_agent


def test_build_aggregated_card_includes_registered_agent(
    registered_echo_agent: str,
) -> None:
    """The aggregated card at /.well-known/agent.json lists every registered agent."""
    card = build_aggregated_card("https://example.test")
    skill_ids = [s["id"] for s in card["skills"]]
    assert registered_echo_agent in skill_ids, (
        f"Aggregated card should surface registered agents; skills: {skill_ids}"
    )
