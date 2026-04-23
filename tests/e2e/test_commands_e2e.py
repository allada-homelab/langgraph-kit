"""End-to-end scenarios for slash-command dispatch.

Commands are intercepted by ``CommandMiddleware.abefore_agent`` before
the LLM is called. A handled command short-circuits via
``jump_to: "end"`` and the dispatcher's output is emitted as an
``AIMessage``. The short-circuit is fragile — it requires both the
``jump_to`` return AND the ``@hook_config(can_jump_to=["end"])``
decorator. A recent bug had the decorator missing, producing a duplicate
response (command output + LLM output). This test locks in the
short-circuit contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from tests.e2e.helpers import (
    last_ai_message,
    scripted_llm,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_slash_compact_short_circuits_without_calling_llm(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """``/compact`` must short-circuit the agent loop — LLM is never invoked.

    Builds a scripted model with zero interactions, so any call to the
    LLM raises ``ReplayMismatchError``. If ``CommandMiddleware``'s
    short-circuit is intact, the dispatcher's output reaches the user
    without the model ever being called and no exception surfaces.

    This guards the CommandMiddleware double-response fix: a previous
    commit (0e21c21) had the ``jump_to: "end"`` return without the
    ``@hook_config(can_jump_to=["end"])`` decorator, so LangChain's
    graph wiring silently ignored the jump and the LLM ran anyway,
    producing a duplicate response. A no-scripted-turns test is the
    cleanest way to detect a regression of that shape: either the LLM
    is never called (test passes) or it is (test raises
    ReplayMismatchError).
    """
    # Zero-turn script: any LLM call raises ReplayMismatchError.
    scripted = scripted_llm([])

    with patched_build_llm(scripted):
        graph, _ = build_deep_agent(
            agent_name="commands-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="/compact")]},
        config={"configurable": {"thread_id": "commands-1"}},  # pyright: ignore[reportArgumentType]
    )

    # The command short-circuit emits an AIMessage with the dispatcher's
    # output. The /compact handler returns either a "Nothing to compact"
    # message (when state is small) or the compacted transcript — both
    # are legitimate evidence that the command ran. The assertion here
    # is just: SOMETHING came back, and it's an AIMessage.
    final = last_ai_message(result)
    assert isinstance(final, AIMessage)
    content = str(final.content)
    assert content, "Command short-circuit produced empty AIMessage content"

    # Signal check: the /compact handler's known outputs all mention
    # either "compact" or "Nothing to compact" (case-insensitive). A
    # different message would indicate the LLM ran and produced its
    # own reply, so this narrows the success criterion.
    low = content.lower()
    assert "compact" in low or "nothing" in low, (
        f"Final AIMessage doesn't look like /compact's output — the "
        f"command short-circuit may be broken and the LLM may have run. "
        f"Got: {content!r}"
    )
