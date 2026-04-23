"""HITL tools — pause agent execution for user approval.

Agents call ``approve_action`` before performing destructive or
irreversible operations. The tool uses LangGraph's ``interrupt()``
to pause the graph; the frontend renders an approval banner and
the user's response is fed back via the ``/resume`` endpoint.
"""

from __future__ import annotations

from typing import Any


def _format_response(response: Any) -> str:
    """Render a HumanResponse-like payload as a human-readable string.

    The resume endpoint feeds back either a ``HumanResponse`` dict or a
    list of them. We return a single string so the agent can reason about
    the outcome without parsing JSON.
    """
    # LangGraph's resume value may be a single response or a list.
    if isinstance(response, list):
        response = response[0] if response else {}

    if not isinstance(response, dict):
        return f"User response: {response!s}"

    rtype = response.get("type")
    args = response.get("args")

    if rtype == "accept":
        return "User accepted the action."
    if rtype == "ignore":
        return "User ignored the action. Do not proceed."
    if rtype == "response":
        msg = args if isinstance(args, str) else str(args or "")
        return (
            f"User rejected the action with message: {msg}"
            if msg
            else "User rejected the action."
        )
    if rtype == "edit":
        if isinstance(args, dict):
            return f"User edited the action. Updated args: {args}"
        return f"User edited the action: {args!s}"

    return f"User response: {response!r}"


def build_approve_action_tool() -> Any:
    """Create the ``approve_action`` tool for agents."""

    # The argument is named ``action_args`` rather than ``args`` because
    # langchain's StructuredTool internal schema already reserves the
    # attribute name ``args`` — passing a tool with a literal ``args``
    # parameter makes LangChain mangle it to ``v__args`` when dispatching,
    # which then fails at call time with:
    #   TypeError: approve_action() got an unexpected keyword argument 'v__args'
    # The interrupt payload still speaks ``{"action_request": {"args": ...}}``
    # so the frontend/resume contract is unchanged.
    async def approve_action(
        action: str,
        description: str,
        action_args: dict[str, Any] | None = None,
    ) -> str:
        """Request user approval before performing a risky action.

        The agent is paused until the user accepts, rejects, or responds.
        Use this before: file deletions, git pushes, external API calls,
        database modifications, or any irreversible operation.

        Args:
            action: Short name for the action (e.g. "delete_file", "git_push")
            description: Human-readable explanation of what will happen
            action_args: Optional details about the action (e.g. {"path": "/etc/config"})
        """
        from langgraph.types import (
            interrupt,  # pyright: ignore[reportMissingModuleSource]
        )

        response = interrupt(
            {
                "action_request": {"action": action, "args": action_args or {}},
                "config": {
                    "allow_ignore": True,
                    "allow_respond": True,
                    "allow_accept": True,
                    "allow_edit": False,
                },
                "description": description,
            }
        )
        return _format_response(response)

    return approve_action
