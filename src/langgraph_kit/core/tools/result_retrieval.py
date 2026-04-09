"""Tool for retrieving persisted large tool results from Store."""

from __future__ import annotations

from typing import Any


def build_result_retrieval_tool(store: Any) -> Any:
    """Create a tool function for retrieving persisted tool results.

    The tool is designed to be passed to create_deep_agent(tools=[...]).
    """

    async def retrieve_result(
        result_ref: str,
        offset: int = 0,
        limit: int = 5000,
    ) -> str:
        """Retrieve a previously persisted large tool result by its reference key.

        Use this when you see a "[Full result persisted — N chars — ref: KEY]"
        message and need to access the complete content.

        Args:
            result_ref: The reference key from the persistence notice
            offset: Character offset to start reading from (default 0)
            limit: Maximum characters to return (default 5000)
        """
        namespace = ("tool_results",)
        item = await store.aget(namespace, result_ref)
        if item is None:
            return f"No persisted result found for ref '{result_ref}'"

        content = item.value.get("content", "")
        tool_name = item.value.get("tool_name", "unknown")
        total = len(content)

        chunk = content[offset : offset + limit]
        remaining = max(0, total - offset - limit)

        header = (
            f"[Retrieved from {tool_name} — "
            f"showing chars {offset}-{offset + len(chunk)} of {total:,}]"
        )
        if remaining > 0:
            header += (
                f"\n[{remaining:,} chars remaining — "
                f"use offset={offset + limit} to continue]"
            )

        return f"{header}\n\n{chunk}"

    return retrieve_result
