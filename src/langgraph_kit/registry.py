"""In-memory registry mapping agent IDs to compiled LangGraph graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_kit.core.commands.dispatch import CommandDispatcher

_registry: dict[str, Any] = {}
_dispatchers: dict[str, CommandDispatcher] = {}


def register(
    agent_id: str,
    graph: Any,
    *,
    command_dispatcher: CommandDispatcher | None = None,
) -> None:
    """Register a compiled graph (and optional command dispatcher) under the given agent ID."""
    _registry[agent_id] = graph
    if command_dispatcher is not None:
        _dispatchers[agent_id] = command_dispatcher


def get(agent_id: str) -> Any:
    """Return the compiled graph for *agent_id*, or raise ``KeyError``."""
    if agent_id not in _registry:
        msg = f"Agent '{agent_id}' not found"
        raise KeyError(msg)
    return _registry[agent_id]


def get_dispatcher(agent_id: str) -> CommandDispatcher | None:
    """Return the command dispatcher for *agent_id*, or None."""
    return _dispatchers.get(agent_id)


def list_agents() -> list[dict[str, str]]:
    """Return metadata for all registered agents."""
    return [
        {"id": agent_id, "name": agent_id.replace("-", " ").title()}
        for agent_id in _registry
    ]
