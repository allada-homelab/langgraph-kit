"""Echo agent — minimal LangGraph StateGraph example.

Demonstrates the standard agent contract:
``build_graph(checkpointer, store) -> CompiledStateGraph``
"""

# NOTE: Do NOT add `from __future__ import annotations` to this file.
# LangGraph evaluates Annotated type hints at runtime via get_type_hints(),
# so annotations must remain as live objects, not strings.

from typing import Any

from langchain_core.runnables import (
    RunnableConfig,  # pyright: ignore[reportMissingModuleSource]
)

from langgraph_kit.llm import build_llm


async def llm_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Call the LLM and return the response message."""
    return {"messages": [await build_llm().ainvoke(state["messages"], config=config)]}


def build_graph(checkpointer: Any, store: Any) -> Any:
    """Build and compile the echo agent graph."""
    from typing import Annotated

    from langgraph.graph import (  # pyright: ignore[reportMissingModuleSource]
        END,
        START,
        StateGraph,
    )
    from langgraph.graph.message import (
        add_messages,  # pyright: ignore[reportMissingModuleSource]
    )
    from typing_extensions import TypedDict

    class AgentState(TypedDict):
        messages: Annotated[list[Any], add_messages]

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile(checkpointer=checkpointer, store=store)
