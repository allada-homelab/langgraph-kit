"""Basic deep agent — minimal deepagents framework wiring.

What this shows
---------------
- ``build_basic_deep_agent(checkpointer, store)`` — the simplest path to
  a working deep agent with the kit's recursion-limit defaults
- The agent has no kit-specific middleware, memory, tools, or
  command dispatcher — see ``reference_deep_agent.py`` for that

How to run
----------
    uv run python -m examples.basic_deep_agent                                  # hermetic
    LANGGRAPH_KIT_EXAMPLES_LLM=real uv run python -m examples.basic_deep_agent  # real LLM

Expected output (hermetic)
--------------------------
    user: Quick check — are you alive?
    assistant: Yes! Basic deep agent is alive and well.
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
    banner("basic_deep_agent")
    user_message = "Quick check — are you alive?"
    line(f"user: {user_message}")

    with tmp_workspace() as workspace:
        if hermetic():
            llm = scripted_llm([answer("Yes! Basic deep agent is alive and well.")])
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

    from langgraph_kit.graphs.basic_deep_agent import build_basic_deep_agent

    _ = workspace

    checkpointer, store = make_in_memory_persistence()
    graph = build_basic_deep_agent(checkpointer, store)

    config = {"configurable": {"thread_id": "demo-basic"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )

    last = result["messages"][-1]
    text = getattr(last, "content", "") or "[no content]"
    line(f"assistant: {text}")


if __name__ == "__main__":
    asyncio.run(main())
