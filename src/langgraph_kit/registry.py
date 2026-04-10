"""In-memory registry mapping agent IDs to compiled LangGraph graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph_kit.core.commands.dispatch import CommandDispatcher


class AgentMetadata(BaseModel):
    """Rich metadata for a registered agent."""

    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


_registry: dict[str, Any] = {}
_dispatchers: dict[str, CommandDispatcher] = {}
_metadata: dict[str, AgentMetadata] = {}


def register(
    agent_id: str,
    graph: Any,
    *,
    command_dispatcher: CommandDispatcher | None = None,
    metadata: AgentMetadata | None = None,
) -> None:
    """Register a compiled graph (and optional command dispatcher) under the given agent ID."""
    _registry[agent_id] = graph
    if command_dispatcher is not None:
        _dispatchers[agent_id] = command_dispatcher
    _metadata[agent_id] = metadata or AgentMetadata()


def get(agent_id: str) -> Any:
    """Return the compiled graph for *agent_id*, or raise ``KeyError``."""
    if agent_id not in _registry:
        msg = f"Agent '{agent_id}' not found"
        raise KeyError(msg)
    return _registry[agent_id]


def get_dispatcher(agent_id: str) -> CommandDispatcher | None:
    """Return the command dispatcher for *agent_id*, or None."""
    return _dispatchers.get(agent_id)


def get_metadata(agent_id: str) -> AgentMetadata:
    """Return the metadata for *agent_id*, or a default instance."""
    return _metadata.get(agent_id, AgentMetadata())


def get_all() -> dict[str, Any]:
    """Return all registered agent graphs keyed by agent ID."""
    return dict(_registry)


def list_agents() -> list[dict[str, Any]]:
    """Return metadata for all registered agents."""
    result: list[dict[str, Any]] = []
    for agent_id in _registry:
        meta = _metadata.get(agent_id, AgentMetadata())
        result.append({
            "id": agent_id,
            "name": agent_id.replace("-", " ").title(),
            "description": meta.description,
            "tags": meta.tags,
            "version": meta.version,
        })
    return result
