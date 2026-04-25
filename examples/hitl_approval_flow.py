"""HITL: interrupt-based approval flow end-to-end.

What this shows
---------------
- Calling :func:`langgraph.types.interrupt` from inside a graph node to
  pause execution and surface a request for user input
- Resuming with :class:`langgraph.types.Command` ``resume=...`` to feed
  the user's decision back into the run
- The kit's :func:`build_approve_action_tool` formatter converting the
  ``{"type": "accept"|"response"|...}`` envelope into a human-readable
  string the agent can reason about

This demo wires a tiny custom graph rather than spinning up a deep
agent, so the interrupt/resume cycle is visible without any LLM in the
loop. The same primitive backs ``ToolCapability(interrupt_before=True)``
when ``AutoInterruptMiddleware`` is wired in.

How to run
----------
    uv run python -m examples.hitl_approval_flow

Expected output
---------------
    --- run 1: initial invoke ---
    Graph paused. Interrupt payload:
      action: delete_branch
      description: Delete branch 'feature/legacy'
    --- run 2: resume with accept ---
    Resolution: User accepted the action.
"""

# NOTE: Do NOT add ``from __future__ import annotations`` here.
# LangGraph's StateGraph runs ``get_type_hints()`` on the State class at
# build time; deferred string annotations would force the eval to look
# up ``add_messages`` in module globals before its import resolves.

from typing import Annotated, Any

from langgraph.graph import (  # pyright: ignore[reportMissingModuleSource]
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import (  # pyright: ignore[reportMissingModuleSource]
    add_messages,
)
from langgraph.types import (  # pyright: ignore[reportMissingModuleSource]
    Command,
    interrupt,
)
from typing_extensions import TypedDict

from examples._lib import banner, line, make_in_memory_persistence
from langgraph_kit.core.hitl.tools import _format_response


class _ApprovalState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    resolution: str


async def _request_approval(_state: _ApprovalState) -> dict[str, Any]:
    """Pause the graph by calling interrupt() with an approval payload."""
    response = interrupt(
        {
            "action_request": {
                "action": "delete_branch",
                "args": {"branch": "feature/legacy"},
            },
            "description": "Delete branch 'feature/legacy'",
        }
    )
    # When the run is resumed, ``response`` is whatever the caller
    # passed via ``Command(resume=...)``. Format it via the kit's bundled
    # helper so the demo matches the agent-facing contract exactly.
    return {"resolution": _format_response(response)}


async def main() -> None:
    import asyncio  # noqa: F401 — silence ruff; the loop runs from __main__

    banner("hitl_approval_flow")

    builder = StateGraph(_ApprovalState)
    builder.add_node(_request_approval)  # pyright: ignore[reportArgumentType]
    builder.add_edge(START, "_request_approval")
    builder.add_edge("_request_approval", END)

    checkpointer, _ = make_in_memory_persistence()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "demo-hitl"}}

    # --- Run 1: initial invoke pauses at the interrupt ----------------
    line("--- run 1: initial invoke ---")
    await graph.ainvoke({"messages": [], "resolution": ""}, config=config)

    state = await graph.aget_state(config)
    interrupts = [
        intr for task in state.tasks for intr in getattr(task, "interrupts", [])
    ]
    if not interrupts:
        line("[unexpected] no interrupts captured")
        return

    payload = interrupts[0].value
    request = payload["action_request"]
    line("Graph paused. Interrupt payload:")
    line(f"  action: {request['action']}")
    line(f"  description: {payload['description']}")

    # --- Run 2: resume with an accept ---------------------------------
    line("--- run 2: resume with accept ---")
    final = await graph.ainvoke(Command(resume={"type": "accept"}), config=config)
    line(f"Resolution: {final['resolution']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
