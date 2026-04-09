"""Slash-command dispatch system.

Transport-independent command dispatch for exposing harness operations
like planning, memory inspection, and context status.
"""

from .dispatch import CommandDispatcher, CommandHandler, CommandInfo, CommandResult
from .middleware import CommandMiddleware

__all__ = [
    "CommandDispatcher",
    "CommandHandler",
    "CommandInfo",
    "CommandMiddleware",
    "CommandResult",
]
