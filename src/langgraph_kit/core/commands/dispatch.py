"""Transport-independent command dispatcher.

Maps slash-command strings to handler functions. Works with any agent,
not just the coding agent.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a dispatched command."""

    output: str
    handled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandInfo:
    """Metadata about a registered command."""

    name: str
    description: str
    usage: str = ""


CommandHandler = Callable[[str, dict[str, Any]], Awaitable[CommandResult]]


class CommandDispatcher:
    """Routes slash-command strings to registered handlers.

    Commands are matched by exact name (case-insensitive). Unrecognized
    commands pass through with ``handled=False``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._commands: dict[str, tuple[CommandHandler, CommandInfo]] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
        usage: str = "",
    ) -> None:
        """Register a command handler."""
        key = name.lower().lstrip("/")
        info = CommandInfo(name=key, description=description, usage=usage)
        self._commands[key] = (handler, info)

    def is_command(self, text: str) -> bool:
        """Check if text starts with / and matches a registered command."""
        if not text.startswith("/"):
            return False
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return False
        return parts[0].lower() in self._commands

    async def dispatch(
        self, text: str, context: dict[str, Any] | None = None
    ) -> CommandResult:
        """Dispatch a slash-command string to the matching handler.

        Returns a ``CommandResult`` with ``handled=False`` if the command
        is not recognized.
        """
        ctx = context or {}
        if not text.startswith("/"):
            return CommandResult(output="", handled=False)

        parts = text[1:].split(maxsplit=1)
        if not parts:
            return CommandResult(output="", handled=False)

        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        entry = self._commands.get(name)
        if entry is None:
            return CommandResult(
                output=f"Unknown command: /{name}",
                handled=False,
            )

        handler, _info = entry
        try:
            return await handler(args, ctx)
        except Exception:
            logger.exception("Command handler failed: /%s", name)
            # ``metadata["error"]`` used to be set here but nothing read
            # it — the middleware only inspected ``output`` and
            # ``handled``. Drop the dead flag; the error prose already
            # conveys the failure state to the user.
            return CommandResult(
                output=f"Error executing /{name}. Check logs for details.",
                handled=True,
            )

    def list_commands(self) -> list[CommandInfo]:
        """Return metadata for all registered commands."""
        return [info for _, info in self._commands.values()]
