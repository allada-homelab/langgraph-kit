"""AutoInterruptMiddleware — capability-driven HITL gating before tool calls.

Pairs with ``ToolCapability.interrupt_before``: when a tool's
capability is registered with ``interrupt_before=True``, this
middleware pauses the graph via LangGraph's ``interrupt()`` *before*
the tool's handler runs. The user's resume payload decides whether to
proceed or skip the tool with an explanatory ``ToolMessage``.

Coexists with the manual ``approve_action`` tool — both can be used
together. Use ``interrupt_before`` for tools whose risk is intrinsic
(e.g. ``delete_file``, ``git_push``); use ``approve_action`` when the
agent should *decide* whether to ask, based on context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from langgraph_kit.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _interpret_resume(response: Any) -> tuple[bool, str]:
    """Map a HumanResponse-like payload to ``(approved, reason)``.

    Mirrors the convention in ``hitl.tools._format_response`` but
    returns a structured outcome the middleware can act on without
    rendering text.
    """
    if isinstance(response, list):
        response = response[0] if response else {}
    if not isinstance(response, dict):
        return (True, "")  # Unknown shape — fail open rather than block work.

    rtype = response.get("type")
    args = response.get("args")

    if rtype == "accept":
        return (True, "")
    if rtype == "ignore":
        return (False, "User chose to skip the action.")
    if rtype == "response":
        msg = args if isinstance(args, str) else str(args or "")
        return (
            False,
            f"User rejected the action: {msg}" if msg else "User rejected the action.",
        )
    if rtype == "edit":
        # We don't currently apply edits to tool args here — caller can
        # use ``approve_action`` for edit-style flows. Treat as a reject
        # with the edited intent surfaced as the reason so the agent can
        # pivot rather than retry the original args.
        return (False, f"User edited the action; original was rejected: {args!r}")

    return (True, "")


class AutoInterruptMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Auto-interrupt before tools whose capability declares
    ``interrupt_before=True``.

    Skips silently if no registry is wired in or the tool has no
    capability registered (in which case the middleware can't know the
    risk profile). Tools without ``interrupt_before`` set proceed
    normally.
    """

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        super().__init__()
        self._tool_registry = tool_registry

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self._tool_registry is None:
            return await handler(request)

        tool_name = request.tool_call.get("name", "")
        cap = self._tool_registry.find_by_tool_name(tool_name)
        if cap is None or not cap.interrupt_before:
            return await handler(request)

        # Lazy import — only this module needs the langgraph runtime API.
        from langgraph.types import (
            interrupt,  # pyright: ignore[reportMissingModuleSource]
        )

        tool_args = request.tool_call.get("args", {}) or {}
        logger.info(
            "AutoInterruptMiddleware: pausing for HITL approval before %s",
            tool_name,
        )
        response = interrupt(
            {
                "action_request": {"action": tool_name, "args": tool_args},
                "config": {
                    "allow_ignore": True,
                    "allow_respond": True,
                    "allow_accept": True,
                    "allow_edit": False,
                },
                "description": (
                    cap.description
                    or f"The agent is about to call `{tool_name}`. Approve to proceed."
                ),
            }
        )
        approved, reason = _interpret_resume(response)
        if approved:
            return await handler(request)

        # Reject path — return a ToolMessage so the agent can adjust
        # rather than crashing the run. Match the failing-tool error
        # shape so models that have seen those messages handle this
        # gracefully.
        tool_call_id = request.tool_call.get("id", "")
        return ToolMessage(
            content=f"Tool call rejected by user. {reason}".strip(),
            tool_call_id=tool_call_id,
            name=tool_name,
        )
