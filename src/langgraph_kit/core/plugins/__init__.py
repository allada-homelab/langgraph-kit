"""Plugin extension system for adding tools, workers, and prompt sections."""

from .loader import PluginLoader
from .mcp import MCPResourceReader, MCPToolAdapter
from .mcp_client import MCPClientManager
from .registry import PluginContribution, PluginRegistry

__all__ = [
    "MCPClientManager",
    "MCPResourceReader",
    "MCPToolAdapter",
    "PluginContribution",
    "PluginLoader",
    "PluginRegistry",
]
