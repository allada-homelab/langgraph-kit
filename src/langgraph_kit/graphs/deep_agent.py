"""Deep agent — deepagents framework example.

Requires the ``deepagents`` package (included in the langgraph
dependency group). If not installed, this agent is skipped
during registration.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.llm import build_llm


def build_deep_graph(checkpointer: Any, store: Any) -> Any:
    """Build and compile a deep agent graph."""
    from deepagents import (
        create_deep_agent,  # pyright: ignore[reportMissingModuleSource]
    )

    return create_deep_agent(model=build_llm(), checkpointer=checkpointer, store=store)
