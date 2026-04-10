"""Tool registration helpers for agent graph builders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph_kit.core.artifacts import build_artifact_tool
from langgraph_kit.core.hitl.tools import build_approve_action_tool
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.orchestration.async_tasks import (
    AsyncTaskManager,
    build_async_task_tools,
)
from langgraph_kit.core.skills.registry import SkillRegistry
from langgraph_kit.core.skills.tools import build_skill_tools
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.deferred import DeferredToolRegistry, build_tool_search
from langgraph_kit.core.tools.memory_tools import build_memory_tools
from langgraph_kit.core.tools.registry import ToolRegistry
from langgraph_kit.core.tools.result_retrieval import build_result_retrieval_tool
from langgraph_kit.core.ui_events import (
    build_citation_tool,
    build_progress_tool,
    build_suggestions_tool,
)

logger = logging.getLogger(__name__)

# Skills directory relative to the package root
_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


def register_tool(
    registry: ToolRegistry,
    tool_fn: Any,
    *,
    id_prefix: str,
    name: str | None = None,
    tags: list[str],
    risk: ToolRisk = ToolRisk.READ_ONLY,
    prompt_guidance: str | None = None,
) -> None:
    """Register a single tool function with metadata."""
    fn_name = name or getattr(tool_fn, "__name__", "tool")
    registry.register(
        ToolCapability(
            id=f"{id_prefix}_{fn_name}" if id_prefix else fn_name,
            name=fn_name,
            description=getattr(tool_fn, "__doc__", "") or "",
            fn=tool_fn,
            tags=tags,
            risk=risk,
            prompt_guidance=prompt_guidance,
        )
    )


def register_memory_tools(
    registry: ToolRegistry, memory_mgr: PersistentMemoryManager
) -> None:
    """Register persistent memory tools (search, save, delete, list)."""
    read_only_names = {"search_memories", "list_memories"}
    for i, tool_fn in enumerate(build_memory_tools(memory_mgr)):
        name = getattr(tool_fn, "__name__", f"memory_tool_{i}")
        register_tool(
            registry,
            tool_fn,
            id_prefix="memory",
            tags=["memory"],
            risk=ToolRisk.READ_ONLY if name in read_only_names else ToolRisk.MUTATING,
            prompt_guidance=(
                "Use memory tools only for stable facts likely to matter in "
                "future sessions. Do not save temporary task state."
            )
            if name == "save_memory"
            else None,
        )


def register_retrieval_tool(registry: ToolRegistry, store: Any) -> None:
    """Register the result retrieval tool for persisted large outputs."""
    register_tool(
        registry,
        build_result_retrieval_tool(store),
        id_prefix="",
        name="retrieve_result",
        tags=["retrieval"],
    )


def register_search_tool(registry: ToolRegistry) -> DeferredToolRegistry:
    """Register the deferred tool search tool. Returns the deferred registry."""
    deferred = DeferredToolRegistry()
    register_tool(
        registry,
        build_tool_search(deferred),
        id_prefix="",
        name="tool_search",
        tags=["discovery"],
    )
    return deferred


def register_skill_tools(
    registry: ToolRegistry, skills_dir: Path | None = None
) -> None:
    """Register skill discovery tools from SKILL.md files."""
    skill_registry = SkillRegistry()
    path = skills_dir or _SKILLS_DIR
    loaded = skill_registry.load_from_directory(path)
    if loaded:
        logger.info("Loaded %d skill(s) from %s", loaded, path)
    for tool_fn in build_skill_tools(skill_registry):
        register_tool(registry, tool_fn, id_prefix="skill", tags=["skills"])


def register_async_tools(
    registry: ToolRegistry, store: Any, *, parent_thread_id: str
) -> None:
    """Register async sub-agent task tools."""
    mgr = AsyncTaskManager(store=store, parent_thread_id=parent_thread_id)
    mutating_names = {"start_async_task", "cancel_async_task"}
    for tool_fn in build_async_task_tools(mgr):
        name = getattr(tool_fn, "__name__", "async_tool")
        register_tool(
            registry,
            tool_fn,
            id_prefix="async",
            tags=["async", "orchestration"],
            risk=ToolRisk.MUTATING if name in mutating_names else ToolRisk.READ_ONLY,
        )


def register_ui_tools(registry: ToolRegistry) -> None:
    """Register artifact, progress, suggestions, and citation tools."""
    register_tool(
        registry,
        build_artifact_tool(),
        id_prefix="",
        name="create_artifact",
        tags=["ui", "artifacts"],
        prompt_guidance=(
            "Use create_artifact for content that benefits from dedicated "
            "rendering: code with syntax highlighting, markdown documents, "
            "data tables, or mermaid diagrams. Do not use for short inline responses."
        ),
    )
    register_tool(
        registry, build_progress_tool(), id_prefix="", name="emit_progress", tags=["ui"]
    )
    register_tool(
        registry,
        build_suggestions_tool(),
        id_prefix="",
        name="suggest_actions",
        tags=["ui"],
    )
    register_tool(
        registry, build_citation_tool(), id_prefix="", name="add_citation", tags=["ui"]
    )


def register_hitl_tools(registry: ToolRegistry) -> None:
    """Register human-in-the-loop approval tools."""
    register_tool(
        registry,
        build_approve_action_tool(),
        id_prefix="",
        name="approve_action",
        tags=["hitl", "approval"],
        risk=ToolRisk.MUTATING,
        prompt_guidance=(
            "Use approve_action before destructive or irreversible operations "
            "like file deletion, git push, database changes, or external API calls. "
            "The user will be shown an approval dialog."
        ),
    )


def register_standard_tools(
    registry: ToolRegistry,
    memory_mgr: PersistentMemoryManager,
    store: Any,
    *,
    parent_thread_id: str,
    mcp_tools: list[Any] | None = None,
) -> None:
    """Register the full standard tool suite used by all deep agents."""
    register_memory_tools(registry, memory_mgr)
    register_retrieval_tool(registry, store)
    register_search_tool(registry)
    register_skill_tools(registry)
    register_async_tools(registry, store, parent_thread_id=parent_thread_id)
    register_ui_tools(registry)
    register_hitl_tools(registry)
    for cap in mcp_tools or []:
        registry.register(cap)
