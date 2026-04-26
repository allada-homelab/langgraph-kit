"""Reference deep agent: the full kit stack wired together.

What this shows
---------------
- Building the canonical deep agent: middleware, memory, tool registry,
  prompt assembly, slash-command dispatcher all wired into one graph
- The agent contract: ``build_reference_deep_agent(checkpointer, store)``
  returns ``(graph, dispatcher)``
- Running a single user turn against it

In hermetic mode the LLM is scripted with one canned answer; the deep
agent's middleware stack still runs (extraction, pressure monitoring,
etc.) but every internal LLM call falls back to fuzzy-matching the same
canned response. That keeps the demo deterministic without claiming to
exercise every code path — for that, see the e2e suite.

How to run
----------
    uv run python -m examples.reference_deep_agent                                  # hermetic
    LANGGRAPH_KIT_EXAMPLES_LLM=real uv run python -m examples.reference_deep_agent  # real LLM

Expected output (hermetic)
--------------------------
    user: Summarise the kit in one line.
    [graph built — middleware: ..., workers: 3, dispatcher commands: 6]
    assistant: A LangGraph deep-agent toolkit with memory, tools, and orchestration.
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
    banner("reference_deep_agent")
    user_message = "Summarise the kit in one line."
    line(f"user: {user_message}")

    with tmp_workspace() as workspace:
        if hermetic():
            llm = scripted_llm(
                [
                    answer(
                        "A LangGraph deep-agent toolkit with memory, tools, and orchestration."
                    )
                ]
            )
            with patch_build_llm(llm):
                await _run(workspace, user_message)
        else:
            assert_real_llm_or_skip()
            configure_real_llm(workspace)
            await _run(workspace, user_message)


async def _run(workspace: object, user_message: str) -> None:
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit.graphs.reference_deep_agent import (
        build_reference_deep_agent,
    )

    _ = workspace

    checkpointer, store = make_in_memory_persistence()
    graph, dispatcher = build_reference_deep_agent(checkpointer, store)
    line(f"[graph built — dispatcher commands: {len(dispatcher.list_commands())}]")

    config = {"configurable": {"thread_id": "demo-reference"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )

    last = result["messages"][-1]
    text = getattr(last, "content", "") or "[no content]"
    line(f"assistant: {text}")


if __name__ == "__main__":
    asyncio.run(main())
