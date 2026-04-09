"""Deferred tool loading — discover and activate tools on demand to reduce prompt bloat."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.tools.capability import ToolCapability


class DeferredToolRegistry:
    """Registry of tools available for on-demand discovery but not loaded by default.

    Deferred tools are NOT included in the agent's initial tool surface.
    The agent discovers them via the tool_search tool and can request activation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tools: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.id] = capability

    def register_many(self, capabilities: list[ToolCapability]) -> None:
        for cap in capabilities:
            self.register(cap)

    def search(self, query: str, limit: int = 5) -> list[ToolCapability]:
        """Search deferred tools by keyword matching against name, description, and tags.

        Simple keyword matching — no vector search needed for a tool catalog.
        """
        query_lower = query.lower()
        scored: list[tuple[int, ToolCapability]] = []

        for cap in self._tools.values():
            score = 0
            if query_lower in cap.name.lower():
                score += 3
            if query_lower in cap.description.lower():
                score += 2
            for tag in cap.tags:
                if query_lower in tag.lower():
                    score += 1
            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cap for _, cap in scored[:limit]]

    def get(self, tool_id: str) -> ToolCapability | None:
        return self._tools.get(tool_id)

    def list_all(self) -> list[ToolCapability]:
        return list(self._tools.values())

    def activate(self, tool_id: str) -> ToolCapability | None:
        """Remove a tool from deferred registry (caller adds it to the active registry)."""
        return self._tools.pop(tool_id, None)


def build_tool_search(deferred: DeferredToolRegistry) -> Any:
    """Create a tool_search tool that agents can use to discover deferred capabilities.

    Returns an async callable suitable for create_deep_agent(tools=[...]).
    """

    async def tool_search(query: str) -> str:
        """Search for additional tools and capabilities not loaded by default.

        Use this when you need a capability that isn't in your current tool set.
        The search checks tool names, descriptions, and tags.

        Args:
            query: Keywords describing the capability you need
        """
        results = deferred.search(query, limit=5)
        if not results:
            return f"No deferred tools found matching '{query}'."

        lines = [f"Found {len(results)} available tool(s):\n"]
        for cap in results:
            tags = f" [{', '.join(cap.tags)}]" if cap.tags else ""
            lines.append(f"- **{cap.name}** (id: {cap.id}){tags}")
            lines.append(f"  {cap.description}")
            if cap.prompt_guidance:
                lines.append(f"  Guidance: {cap.prompt_guidance}")
            lines.append("")
        lines.append(
            "To use a tool, call it by name — it will be activated automatically."
        )
        return "\n".join(lines)

    return tool_search
