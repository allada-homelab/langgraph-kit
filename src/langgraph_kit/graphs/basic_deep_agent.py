"""Basic deep agent — minimal deepagents framework example.

Framework defaults plus a generic system prompt and a graph name for
tracing. No kit features are wired in (no persistent memory, custom tool
registry, middleware stack, workers, or slash commands). See
``reference_deep_agent`` for the full-featured version to clone from when
starting a new domain agent.

Requires the ``deepagents`` package (included in the langgraph
dependency group). If not installed, this agent is skipped
during registration.

The built graph is bound to :data:`~langgraph_kit.graphs._builder.DEFAULT_RECURSION_LIMIT`
(100). Override with ``recursion_limit=<n>`` here or
``config={"recursion_limit": <n>}`` at invoke time.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.graphs._basic_prompt import BASIC_SYSTEM_PROMPT
from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT, bind_kit_defaults
from langgraph_kit.llm import build_llm


def build_basic_deep_agent(
    checkpointer: Any,
    store: Any,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> Any:
    """Build and compile a basic deep agent graph.

    Parameters
    ----------
    recursion_limit:
        Default LangGraph recursion limit bound to the compiled graph.
        Defaults to :data:`DEFAULT_RECURSION_LIMIT` (100). Pass a higher
        value for long-running runs, or override per-invocation via
        ``config={"recursion_limit": N}``.
    """
    from deepagents import (
        create_deep_agent,  # pyright: ignore[reportMissingModuleSource]
    )

    graph = create_deep_agent(
        model=build_llm(),
        system_prompt=BASIC_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        name="basic-deep-agent",
    )
    return bind_kit_defaults(graph, recursion_limit=recursion_limit)
