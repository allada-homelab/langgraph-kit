"""Backend factory for deepagents-based graph builders."""

from __future__ import annotations

from typing import Any


def build_backend_factory(agent_name: str) -> Any:
    """Create a CompositeBackend factory with agent-specific namespaces.

    Routes:
      /memories/  -> StoreBackend (persistent, namespaced per-agent)
      /notes/     -> StoreBackend (session notes, namespaced per-thread)
      default     -> StateBackend (ephemeral per-thread scratch)
    """
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    def factory(runtime: Any) -> Any:
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(
                    runtime, namespace=lambda _ctx: (agent_name, "memories")
                ),
                "/notes/": StoreBackend(
                    runtime, namespace=lambda _ctx: (agent_name, "notes")
                ),
            },
        )

    return factory
