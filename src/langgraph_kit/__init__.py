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

from langgraph_kit._config import (
    AgentConfig,
    configure,
    configure_from_settings,
    get_config,
)
from langgraph_kit.cancellation import (
    ThreadCancellationRegistry,
    cancel_thread,
    get_cancellation_registry,
)
from langgraph_kit.core.coordinator import COORDINATOR_SECTIONS, CoordinatorMode
from langgraph_kit.core.memory.session import SessionNotebook
from langgraph_kit.llm import build_llm
from langgraph_kit.models import ChatMessage, InvokeRequest, InvokeResponse
from langgraph_kit.observability import UserInfo, build_agent_run_config
from langgraph_kit.persistence import create_persistence
from langgraph_kit.registry import (
    AgentMetadata,
    get,
    get_all,
    get_dispatcher,
    get_metadata,
    list_agents,
    register,
)
from langgraph_kit.streaming import stream_agent_events

__all__ = [
    "COORDINATOR_SECTIONS",
    "AgentConfig",
    "AgentMetadata",
    "ChatMessage",
    "CoordinatorMode",
    "InvokeRequest",
    "InvokeResponse",
    "SessionNotebook",
    "ThreadCancellationRegistry",
    "UserInfo",
    "build_agent_run_config",
    "build_llm",
    "cancel_thread",
    "configure",
    "configure_from_settings",
    "create_persistence",
    "get",
    "get_all",
    "get_cancellation_registry",
    "get_config",
    "get_dispatcher",
    "get_metadata",
    "list_agents",
    "register",
    "stream_agent_events",
]
