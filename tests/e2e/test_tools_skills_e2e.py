"""Cluster A — end-to-end skill discovery + loading flows.

``list_skills()`` → discovery; ``read_skill(name)`` → load full
instructions. Together they implement the progressive-disclosure
pattern the kit relies on for bundling specialized workflows without
bloating every prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_list_skills_then_read_skill_full_flow(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """LLM discovers skills via list_skills, then loads one via read_skill.

    The kit ships two default skills (``code-review`` and ``research``)
    under ``src/langgraph_kit/skills/``. ``register_skill_tools`` in
    the standard tool bundle loads them at build. This test exercises
    the full round-trip: list returns at least one skill by name; read
    returns non-empty content that isn't the "not found" error path.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("list_skills"),
            tool_call_turn("read_skill", {"name": "code-review"}),
            answer("loaded"),
        ]
    )

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="skills-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="what skills are available?")]},
        config={"configurable": {"thread_id": "skills-1"}},  # pyright: ignore[reportArgumentType]
    )

    list_msg = assert_tool_invoked(result, "list_skills")
    list_content = str(list_msg.content)
    assert "code-review" in list_content or "research" in list_content, (
        f"list_skills returned unexpected output (are default skills loading?): "
        f"{list_content!r}"
    )

    read_msg = assert_tool_invoked(result, "read_skill")
    read_content = str(read_msg.content)
    assert "not found" not in read_content.lower(), (
        f"read_skill couldn't find a skill that list_skills said exists: "
        f"{read_content!r}"
    )
    # Distinct skill content is usually multi-paragraph. A not-found
    # error is typically short. This is a rough but effective sanity.
    assert len(read_content) > 100, (
        f"read_skill returned suspiciously short content: {read_content!r}"
    )


@pytest.mark.asyncio
async def test_read_skill_unknown_name_returns_not_found(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """``read_skill`` with an unknown name must surface a recoverable error.

    The tool returns ``"Skill '...' not found. Available: ..."`` as a
    string so the LLM can adjust on its next turn. A ValueError raise
    would crash the run; this locks in the recoverable-error contract.
    """
    scripted = scripted_llm(
        [
            tool_call_turn("read_skill", {"name": "does-not-exist"}),
            answer("handled"),
        ]
    )

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="skills-unknown",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="load a skill")]},
        config={"configurable": {"thread_id": "skills-unknown"}},  # pyright: ignore[reportArgumentType]
    )

    msg = assert_tool_invoked(result, "read_skill")
    content = str(msg.content)
    assert "not found" in content.lower(), (
        f"Expected not-found error; got: {content!r}"
    )
    assert "Available" in content, (
        f"Not-found error should list available skills; got: {content!r}"
    )
