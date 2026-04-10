"""Shared agent builder utilities — re-exports from core.graph_builder.

This module exists for backwards compatibility. New code should import
directly from ``langgraph_kit.core.graph_builder``.
"""

from langgraph_kit.core.graph_builder.backend import build_backend_factory
from langgraph_kit.core.graph_builder.commands import build_command_dispatcher
from langgraph_kit.core.graph_builder.middleware import build_middleware_stack
from langgraph_kit.core.graph_builder.tools import (
    register_standard_tools,
    register_tool,
)

# Backwards-compatible alias
_register_tool = register_tool

__all__ = [
    "_register_tool",
    "build_backend_factory",
    "build_command_dispatcher",
    "build_middleware_stack",
    "register_standard_tools",
    "register_tool",
]
