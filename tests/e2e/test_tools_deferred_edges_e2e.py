"""Cluster A edges — ``call_deferred_tool`` error paths.

The happy-path flow is covered in ``test_deferred_tools_e2e.py``. This
file covers the structured-error surface the dispatcher produces when
the LLM sends malformed arguments — every branch of
``call_deferred_tool`` that returns a recoverable error string rather
than raising. Catching these as structured results (and not
exceptions) is what lets the agent retry instead of dying.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    answer,
    assert_tool_invoked,
    scripted_llm,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e


async def greet(name: str) -> str:
    """Simple deferred tool body that takes one string argument."""
    return f"HELLO {name.upper()}"


def _populate_greet(registry: Any) -> None:
    registry.register(
        ToolCapability(
            id="greet",
            name="greet",
            description="Greet a user by name.",
            fn=greet,
            risk=ToolRisk.READ_ONLY,
        )
    )


@pytest.mark.asyncio
async def test_call_deferred_tool_unknown_id_returns_sentinel(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Unknown ``tool_id`` → recoverable 'not found' error with id hints."""
    scripted = scripted_llm(
        [
            tool_call_turn(
                "call_deferred_tool",
                {"tool_id": "nonexistent-tool", "arguments": {}},
            ),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="deferred-unknown-id",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_deferred_tools=_populate_greet,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="do it")]},
        config={"configurable": {"thread_id": "deferred-unknown"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "call_deferred_tool")
    content = str(tool_msg.content).lower()
    assert "not found" in content, (
        f"Unknown tool_id should surface 'not found'; got {tool_msg.content!r}"
    )
    # The dispatcher includes up to 10 available ids so the model can
    # recover without a second tool_search roundtrip.
    assert "greet" in content, (
        "Error should name available ids so the model can correct itself"
    )


@pytest.mark.asyncio
async def test_call_deferred_tool_wrong_arg_shape_returns_typeerror_as_string(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """Wrong argument shape → ``TypeError`` surfaces as a recoverable error."""
    # greet(name: str) — calling it with no ``name`` should TypeError.
    scripted = scripted_llm(
        [
            tool_call_turn(
                "call_deferred_tool",
                {"tool_id": "greet", "arguments": {"wrong_key": "Alice"}},
            ),
            answer("noted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="deferred-wrong-shape",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_deferred_tools=_populate_greet,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="do it")]},
        config={"configurable": {"thread_id": "deferred-shape"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "call_deferred_tool")
    content = str(tool_msg.content)
    assert "Error calling 'greet'" in content, (
        f"Wrong-shape call should surface 'Error calling ...'; got {content!r}"
    )
    # TypeError text should be included so the model can self-correct.
    assert "argument" in content.lower() or "wrong_key" in content, (
        f"Error should describe the argument mismatch; got {content!r}"
    )


@pytest.mark.asyncio
async def test_call_deferred_tool_json_string_arguments_are_parsed_and_dispatched(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """LLM-emitted JSON-string ``arguments`` → parsed and forwarded to the tool body.

    Some models (Qwen variants, notably) emit the ``arguments`` field of
    a tool call as a JSON-encoded string instead of an object. The
    dispatcher's signature admits ``dict | str | None`` so LangChain's
    ``StructuredTool`` lets the string through pydantic; the in-body
    ``json.loads`` fallback then unwraps it and dispatches normally.
    Without this path the agent loops retrying the same malformed shape
    because the rejection looks identical across retries.
    """
    scripted = scripted_llm(
        [
            tool_call_turn(
                "call_deferred_tool",
                {"tool_id": "greet", "arguments": '{"name": "Alice"}'},
            ),
            answer("greeted"),
        ]
    )
    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="deferred-json-str",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
            configure_deferred_tools=_populate_greet,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="greet Alice")]},
        config={"configurable": {"thread_id": "deferred-json"}},  # pyright: ignore[reportArgumentType]
    )

    tool_msg = assert_tool_invoked(result, "call_deferred_tool")
    content = str(tool_msg.content)
    assert content == "HELLO ALICE", (
        f"JSON-string arguments should be parsed and forwarded to the tool;"
        f" got {content!r}"
    )
