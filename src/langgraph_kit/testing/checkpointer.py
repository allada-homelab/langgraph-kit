"""In-memory checkpointer with testing-friendly affordances.

A thin wrapper around LangGraph's :class:`InMemorySaver` that adds a
small assertion / introspection surface for tests. Use as a drop-in
replacement for ``InMemorySaver``: every kit module that takes a
``checkpointer`` accepts this unchanged.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import (  # pyright: ignore[reportMissingModuleSource]
    RunnableConfig,
)
from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
    InMemorySaver,
)


class FakeCheckpointer(InMemorySaver):
    """In-memory checkpointer with ``dump_state`` / assertion helpers.

    All persistence goes through :class:`InMemorySaver` so every code
    path the kit tests against the real saver runs unchanged here. The
    extra methods only read state, never alter it.
    """

    def dump_state(self, thread_id: str) -> dict[str, Any]:
        """Return the latest checkpoint values for *thread_id*.

        Empty dict if the thread has no checkpoints. Useful for tests
        that need to inspect what the graph wrote without round-
        tripping through ``aget_state`` / ``StateSnapshot`` shapes.

        Plain-``dict`` state graphs nest the whole state under the
        synthetic ``__root__`` channel; ``TypedDict`` state graphs
        expose each field as its own channel. This unwraps the former
        so callers see the same shape in both cases.
        """
        config = cast("RunnableConfig", {"configurable": {"thread_id": thread_id}})
        checkpoint = self.get(config)
        if checkpoint is None:
            return {}
        values = checkpoint.get("channel_values", {})
        if not values:
            return {}
        root = values.get("__root__")
        if isinstance(root, dict) and len(values) == 1:
            return dict(root)
        return dict(values)

    def assert_thread_has_messages(self, thread_id: str, n: int) -> None:
        """Assert *thread_id*'s message list has exactly *n* entries.

        Raises :class:`AssertionError` with the actual count and a
        truncated summary of message types when the count differs.
        """
        values = self.dump_state(thread_id)
        messages = values.get("messages", [])
        actual = len(messages)
        if actual != n:
            summary = [type(m).__name__ for m in messages[:5]]
            suffix = "..." if actual > 5 else ""
            raise AssertionError(
                f"Expected thread {thread_id!r} to have {n} messages; "
                f"got {actual} (types: {summary}{suffix})"
            )


__all__ = ["FakeCheckpointer"]
