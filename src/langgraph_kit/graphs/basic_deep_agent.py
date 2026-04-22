"""Basic deep agent — minimal deepagents framework example.

Framework defaults plus a generic system prompt and a graph name for
tracing. No kit features are wired in (no persistent memory, custom tool
registry, middleware stack, workers, or slash commands). See
``reference_deep_agent`` for the full-featured version to clone from when
starting a new domain agent.

Requires the ``deepagents`` package (included in the langgraph
dependency group). If not installed, this agent is skipped
during registration.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.graphs._basic_prompt import BASIC_SYSTEM_PROMPT
from langgraph_kit.llm import build_llm


def build_basic_deep_agent(checkpointer: Any, store: Any) -> Any:
    """Build and compile a basic deep agent graph."""
    from deepagents import (
        create_deep_agent,  # pyright: ignore[reportMissingModuleSource]
    )

    return create_deep_agent(
        model=build_llm(),
        system_prompt=BASIC_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        name="basic-deep-agent",
    )
