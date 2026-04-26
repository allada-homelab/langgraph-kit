"""Quickstart: echo agent, end-to-end in 30 lines.

What this shows
---------------
- The minimal langgraph-kit agent contract: ``build_graph(checkpointer, store)``
- Hermetic execution via the kit's ``RecordedChatModel`` substrate
- Threaded conversation state via the LangGraph checkpointer

How to run
----------
    uv run python -m examples.quickstart_echo                                  # hermetic (default)
    LANGGRAPH_KIT_EXAMPLES_LLM=real uv run python -m examples.quickstart_echo  # real LLM (needs AGENT_LLM_API_KEY)

Expected output (hermetic)
--------------------------
    user: Hello, kit!
    assistant: Hi there — this is the echo agent saying hello back.
"""

from __future__ import annotations

import asyncio

from examples._lib import (
    answer,
    assert_real_llm_or_skip,
    banner,
    configure_real_llm,
    hermetic,
    line,
    make_in_memory_persistence,
    patch_build_llm,
    scripted_llm,
    tmp_workspace,
)


async def main() -> None:
    banner("quickstart_echo")
    user_message = "Hello, kit!"
    line(f"user: {user_message}")

    with tmp_workspace() as workspace:
        if hermetic():
            llm = scripted_llm(
                [answer("Hi there — this is the echo agent saying hello back.")]
            )
            with patch_build_llm(llm):
                await _run_one_turn(workspace, user_message)
        else:
            assert_real_llm_or_skip()
            configure_real_llm(workspace)
            await _run_one_turn(workspace, user_message)


async def _run_one_turn(workspace: object, user_message: str) -> None:
    """Build the echo agent, send one user message, print the reply."""
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit.graphs.echo_agent import build_graph

    _ = workspace  # echo agent doesn't persist anything outside the in-memory store

    checkpointer, store = make_in_memory_persistence()
    graph = build_graph(checkpointer, store)
    config = {"configurable": {"thread_id": "demo-thread"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )

    last = result["messages"][-1]
    line(f"assistant: {last.content}")


if __name__ == "__main__":
    asyncio.run(main())
