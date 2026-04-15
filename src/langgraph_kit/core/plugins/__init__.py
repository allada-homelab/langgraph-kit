"""Plugin extension system for adding tools, workers, and prompt sections."""

from .loader import PluginLoader
from .mcp import adapt_mcp_tool, adapt_mcp_tools
from .mcp_client import MCPClientManager
from .registry import PluginContribution, PluginRegistry

__all__ = [
    "MCPClientManager",
    "adapt_mcp_tool",
    "adapt_mcp_tools",
    "PluginContribution",
    "PluginLoader",
    "PluginRegistry",
]
