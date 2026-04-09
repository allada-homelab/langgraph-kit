"""Reusable LangGraph agent toolkit.

Quick start::

    from langgraph_kit import AgentConfig, configure
    configure(AgentConfig(llm_model="gpt-4o", database_url="postgresql://..."))

    from langgraph_kit import build_llm, create_persistence
    from langgraph_kit.graphs import register_all

Subpackages:
    core      Building blocks (memory, tools, commands, resilience, ...)
    graphs    Agent graph definitions (echo, deep, r0, coding)
    contrib   Optional integrations (FastAPI router factory)
    evals     Evaluation framework for agent quality
"""

from langgraph_kit._config import AgentConfig, configure, get_config
from langgraph_kit.llm import build_llm
from langgraph_kit.models import ChatMessage, InvokeRequest, InvokeResponse
from langgraph_kit.observability import UserInfo, build_agent_run_config
from langgraph_kit.persistence import create_persistence
from langgraph_kit.registry import get, get_dispatcher, list_agents, register
from langgraph_kit.streaming import stream_agent_events

__all__ = [
    "AgentConfig",
    "ChatMessage",
    "InvokeRequest",
    "InvokeResponse",
    "UserInfo",
    "build_agent_run_config",
    "build_llm",
    "configure",
    "create_persistence",
    "get",
    "get_config",
    "get_dispatcher",
    "list_agents",
    "register",
    "stream_agent_events",
]
