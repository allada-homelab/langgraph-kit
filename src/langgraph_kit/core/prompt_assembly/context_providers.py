"""Dynamic context providers that inject runtime information into prompts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContextProvider(Protocol):
    async def provide(self, context: dict[str, Any]) -> str: ...


class ThreadContextProvider:
    """Injects thread metadata into the prompt."""

    async def provide(self, context: dict[str, Any]) -> str:
        thread_id = context.get("thread_id", "unknown")
        message_count = context.get("message_count", 0)
        return f"Thread: {thread_id}\nMessages: {message_count}"


class MemoryContextProvider:
    """Injects relevant memory summaries into the prompt."""

    def __init__(self, memories: list[str] | None = None) -> None:
        super().__init__()
        self._memories = memories

    async def provide(self, context: dict[str, Any]) -> str:
        memories: list[str] = context.get("memories", self._memories or [])
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return f"# Relevant Memory\n{lines}"


class ToolContextProvider:
    """Injects tool usage guidance into the prompt."""

    def __init__(self, tool_guidance: list[str] | None = None) -> None:
        super().__init__()
        self._tool_guidance = tool_guidance

    async def provide(self, context: dict[str, Any]) -> str:
        guidance: list[str] = context.get("tool_guidance", self._tool_guidance or [])
        if not guidance:
            return ""
        lines = "\n".join(f"- {g}" for g in guidance)
        return f"# Tool Guidance\n{lines}"
