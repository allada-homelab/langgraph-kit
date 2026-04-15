"""Plugin extension system for adding tools, workers, and prompt sections."""

from .loader import PluginLoader
from .mcp import MCPToolAdapter
from .mcp_client import MCPClientManager
from .registry import PluginContribution, PluginRegistry

__all__ = [
    "MCPClientManager",
    "MCPToolAdapter",
    "PluginContribution",
    "PluginLoader",
    "PluginRegistry",
]
