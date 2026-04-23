"""Smoke test — validates Phase 2 infrastructure composes end-to-end.

Throwaway: covers the same ground as Phase 1's spike but goes through
the conftest + helpers, proving they wire together. Delete once Phase 3
scenarios are landing (they re-verify the same ground implicitly).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs.reference_deep_agent import build_reference_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    last_ai_message,
    scripted_llm,
    tool_call_turn,
)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_infrastructure_smoke(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Helpers + fixtures drive a real graph through a 2-turn conversation."""
    scripted = scripted_llm(
        [
            tool_call_turn("list_skills"),
            answer("smoke"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_reference_deep_agent(
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="go")]},
        config={"configurable": {"thread_id": "smoke-1"}},  # pyright: ignore[reportArgumentType]
    )

    assert_tool_invoked(result, "list_skills")
    assert "smoke" in last_ai_message(result).content
