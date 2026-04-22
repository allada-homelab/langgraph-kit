# pyright: reportPrivateUsage=false
# Strict-mode prep: these tests reach into the registry's private
# ``_metadata`` / ``_registry`` / ``_dispatchers`` dicts to set up and
# clean state between cases.  Keeps the file clean under a future
# strict flip without weakening type safety elsewhere.
"""Tests for the agent registry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langgraph_kit.registry import (
    AgentMetadata,
    _dispatchers,
    _metadata,
    _registry,
    get,
    get_all,
    get_dispatcher,
    get_metadata,
    list_agents,
    register,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear registry state before and after each test."""
    _registry.clear()
    _dispatchers.clear()
    _metadata.clear()
    yield
    _registry.clear()
    _dispatchers.clear()
    _metadata.clear()


class TestRegister:
    def test_register_graph(self) -> None:
        graph = MagicMock(name="my_graph")
        register("echo", graph)
        assert get("echo") is graph

    def test_register_with_dispatcher(self) -> None:
        graph = MagicMock()
        dispatcher = MagicMock()
        register("echo", graph, command_dispatcher=dispatcher)
        assert get_dispatcher("echo") is dispatcher

    def test_register_with_metadata(self) -> None:
        graph = MagicMock()
        meta = AgentMetadata(description="Test agent", version="2.0.0")
        register("echo", graph, metadata=meta)
        assert get_metadata("echo").description == "Test agent"
        assert get_metadata("echo").version == "2.0.0"

    def test_register_default_metadata(self) -> None:
        register("echo", MagicMock())
        meta = get_metadata("echo")
        assert meta.description == ""
        assert meta.version == "1.0.0"


class TestGet:
    def test_get_missing_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get("nonexistent")

    def test_get_registered(self) -> None:
        graph = MagicMock()
        register("test", graph)
        assert get("test") is graph


class TestGetDispatcher:
    def test_returns_none_when_unset(self) -> None:
        register("test", MagicMock())
        assert get_dispatcher("test") is None

    def test_returns_none_for_unknown(self) -> None:
        assert get_dispatcher("unknown") is None


class TestGetMetadata:
    def test_returns_default_for_unknown(self) -> None:
        meta = get_metadata("unknown")
        assert isinstance(meta, AgentMetadata)
        assert meta.description == ""


class TestGetAll:
    def test_empty(self) -> None:
        assert get_all() == {}

    def test_returns_copy(self) -> None:
        register("a", MagicMock())
        register("b", MagicMock())
        all_agents = get_all()
        assert len(all_agents) == 2
        assert "a" in all_agents
        assert "b" in all_agents


class TestListAgents:
    def test_empty(self) -> None:
        assert list_agents() == []

    def test_formats_agent_name(self) -> None:
        register("my-cool-agent", MagicMock())
        agents = list_agents()
        assert len(agents) == 1
        assert agents[0]["id"] == "my-cool-agent"
        assert agents[0]["name"] == "My Cool Agent"

    def test_includes_metadata(self) -> None:
        meta = AgentMetadata(description="A test", tags=["demo"], version="3.0")
        register("test", MagicMock(), metadata=meta)
        agents = list_agents()
        assert agents[0]["description"] == "A test"
        assert agents[0]["tags"] == ["demo"]
        assert agents[0]["version"] == "3.0"
