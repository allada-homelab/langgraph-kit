"""HITL tools — pause agent execution for user approval.

Agents call ``approve_action`` before performing destructive or
irreversible operations. The tool uses LangGraph's ``interrupt()``
to pause the graph; the frontend renders an approval banner and
the user's response is fed back via the ``/resume`` endpoint.
"""

from __future__ import annotations

import json
from typing import Any


def build_approve_action_tool() -> Any:
    """Create the ``approve_action`` tool for agents."""

    async def approve_action(
        action: str,
        description: str,
        args: dict[str, Any] | None = None,
    ) -> str:
        """Request user approval before performing a risky action.

        The agent is paused until the user accepts, rejects, or responds.
        Use this before: file deletions, git pushes, external API calls,
        database modifications, or any irreversible operation.

        Args:
            action: Short name for the action (e.g. "delete_file", "git_push")
            description: Human-readable explanation of what will happen
            args: Optional details about the action (e.g. {"path": "/etc/config"})
        """
        from langgraph.types import (
            interrupt,  # pyright: ignore[reportMissingModuleSource]
        )

        response = interrupt(
            {
                "action_request": {"action": action, "args": args or {}},
                "config": {
                    "allow_ignore": True,
                    "allow_respond": True,
                    "allow_accept": True,
                    "allow_edit": False,
                },
                "description": description,
            }
        )
        return json.dumps(response)

    return approve_action
