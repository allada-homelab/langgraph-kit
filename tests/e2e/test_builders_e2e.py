"""Cluster H — builder-surface smoke tests.

Each kit-provided builder should drive a real graph end-to-end with a
scripted LLM. These tests are short smoke scenarios — the deep
feature coverage lives in the cluster-specific test files — the goal
here is "does this builder even compose?" for every public builder.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs.basic_deep_agent import build_basic_deep_agent
from langgraph_kit.graphs.coding_agent import build_coding_agent
from tests.e2e.helpers import answer, last_ai_message, scripted_llm

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_build_basic_deep_agent_smoke(checkpointer: Any, e2e_store: Any) -> None:
    """``build_basic_deep_agent`` produces a working graph end-to-end.

    Basic deep agent has no kit middleware, memory, or tools — just a
    model + system prompt. This test verifies the graph compiles and
    ``ainvoke`` produces the scripted response.

    Note: basic_deep_agent imports ``build_llm`` from
    ``langgraph_kit.llm`` directly (not via ``_builder``), so the
    standard ``patched_build_llm`` fixture targets the wrong reference.
    We patch at the import site here instead.
    """
    scripted = scripted_llm([answer("basic-agent-ok")])

    with patch(
        "langgraph_kit.graphs.basic_deep_agent.build_llm",
        return_value=scripted,
    ):
        graph = build_basic_deep_agent(
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "basic-smoke"}},  # pyright: ignore[reportArgumentType]
    )
    final = last_ai_message(result)
    assert "basic-agent-ok" in str(final.content)


@pytest.mark.asyncio
async def test_build_coding_agent_smoke(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """``build_coding_agent`` produces a working graph end-to-end.

    Coding agent layers coding-profile prompt sections, a
    ``GitContextProvider``, worktree tools, and a stricter verification
    worker over the reference skeleton. Smoke: the scripted LLM reaches
    the graph and its answer comes back unchanged.

    ``build_coding_agent`` routes through ``build_deep_agent``, which
    resolves ``build_llm`` from ``langgraph_kit.graphs._builder`` — so
    the standard ``patched_build_llm`` fixture applies here (unlike
    ``build_basic_deep_agent`` above).
    """
    scripted = scripted_llm([answer("coding-agent-ok")])
    with patched_build_llm(scripted):
        graph, _dispatcher = build_coding_agent(
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="look at the repo")]},
        config={"configurable": {"thread_id": "coding-smoke"}},  # pyright: ignore[reportArgumentType]
    )
    final = last_ai_message(result)
    assert "coding-agent-ok" in str(final.content), (
        f"Coding agent didn't surface the scripted response; got {final.content!r}"
    )
