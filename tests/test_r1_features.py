"""Tests for R1 LangGraph agent modules."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph_kit.core.context_management.result_persistence import (
    ResultPersistenceMiddleware,
)
from langgraph_kit.core.coordinator import (
    COORDINATOR_SECTIONS,
    CoordinatorMode,
)
from langgraph_kit.core.memory.agent_memory import AgentMemoryManager
from langgraph_kit.core.memory.consolidation import (
    MemoryConsolidator,
)
from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.memory.shared import (
    SecretDetectedError,
    SharedMemoryManager,
)
from langgraph_kit.core.plugins.mcp import adapt_mcp_tool, adapt_mcp_tools
from langgraph_kit.core.plugins.registry import (
    PluginContribution,
    PluginRegistry,
)
from langgraph_kit.core.prompt_assembly.activation import (
    ACTIVATION_SECTIONS,
)
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.deferred import (
    DeferredToolRegistry,
    build_tool_search,
)

# ---------------------------------------------------------------------------
# 1. ResultPersistenceMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_small_result_passes_through() -> None:
    middleware = ResultPersistenceMiddleware(persist_threshold=4000)

    request = MagicMock()
    request.tool_call = {"name": "grep", "id": "call_1"}

    result_msg = MagicMock()
    result_msg.content = "short"

    handler = AsyncMock(return_value=result_msg)
    result = await middleware.awrap_tool_call(request, handler)

    assert result.content == "short"


@pytest.mark.asyncio
async def test_large_result_persisted_and_replaced(mock_store: Any) -> None:
    middleware = ResultPersistenceMiddleware(persist_threshold=4000, preview_length=300)

    request = MagicMock()
    request.tool_call = {"name": "read_file", "id": "call_2"}
    request.runtime.store = mock_store

    large_content = "x" * 5000
    result_msg = MagicMock()
    result_msg.content = large_content
    result_msg.model_copy = lambda *, update: MagicMock(content=update["content"])

    handler = AsyncMock(return_value=result_msg)
    result = await middleware.awrap_tool_call(request, handler)

    assert "[Full result persisted" in result.content
    assert "5,000 chars" in result.content

    # Verify the store received the data
    stored = mock_store._data.get(("tool_results",))
    assert stored is not None
    assert len(stored) == 1
    stored_val = next(iter(stored.values()))
    assert stored_val["content"] == large_content
    assert stored_val["tool_name"] == "read_file"


# ---------------------------------------------------------------------------
# 2. DeferredToolRegistry + build_tool_search
# ---------------------------------------------------------------------------


def _make_cap(
    cap_id: str, name: str, description: str = "", tags: list[str] | None = None
) -> ToolCapability:
    return ToolCapability(
        id=cap_id,
        name=name,
        description=description or f"Description for {name}",
        fn=lambda: None,
        tags=tags or [],
        risk=ToolRisk.READ_ONLY,
    )


def test_deferred_search_by_name() -> None:
    registry = DeferredToolRegistry()
    registry.register(_make_cap("fs", "file_search", "Search files on disk"))
    results = registry.search("file")
    assert len(results) == 1
    assert results[0].name == "file_search"


def test_deferred_search_no_match() -> None:
    registry = DeferredToolRegistry()
    registry.register(_make_cap("fs", "file_search"))
    results = registry.search("nonexistent")
    assert results == []


def test_deferred_activate() -> None:
    registry = DeferredToolRegistry()
    cap = _make_cap("fs", "file_search")
    registry.register(cap)

    activated = registry.activate("fs")
    assert activated is not None
    assert activated.id == "fs"

    # Should be removed from deferred
    assert registry.get("fs") is None
    assert registry.list_all() == []


@pytest.mark.asyncio
async def test_tool_search_returns_formatted() -> None:
    registry = DeferredToolRegistry()
    registry.register(
        _make_cap("fs", "file_search", "Search files", tags=["filesystem"])
    )

    tool_search = build_tool_search(registry)
    output = await tool_search("file")

    assert "Found 1 available tool(s)" in output
    assert "file_search" in output
    assert "Search files" in output


# ---------------------------------------------------------------------------
# 3. AgentMemoryManager
# ---------------------------------------------------------------------------


def _make_record(
    title: str = "Test",
    memory_type: MemoryType = MemoryType.PROJECT,
    scope: MemoryScope = MemoryScope.USER,
    body: str = "body",
) -> MemoryRecord:
    return MemoryRecord(
        title=title, type=memory_type, scope=scope, summary="summary", body=body
    )


@pytest.mark.asyncio
async def test_agent_memory_create_and_get(mock_store: Any) -> None:
    mgr = AgentMemoryManager(mock_store, agent_name="researcher")
    record = _make_record(title="Pattern A")
    await mgr.create(record)

    retrieved = await mgr.get(record.id, memory_type=record.type)
    assert retrieved is not None
    assert retrieved.title == "Pattern A"


@pytest.mark.asyncio
async def test_agent_memory_list_all(mock_store: Any) -> None:
    mgr = AgentMemoryManager(mock_store, agent_name="researcher")
    await mgr.create(_make_record(title="Record 1"))
    await mgr.create(_make_record(title="Record 2"))

    records = await mgr.list_all()
    assert len(records) == 2


@pytest.mark.asyncio
async def test_agent_memory_namespaced_separately(mock_store: Any) -> None:
    researcher_mgr = AgentMemoryManager(mock_store, agent_name="researcher")
    implementer_mgr = AgentMemoryManager(mock_store, agent_name="implementer")

    record = _make_record(title="Researcher only")
    await researcher_mgr.create(record)

    # Should not appear in implementer's namespace
    implementer_records = await implementer_mgr.list_all()
    assert len(implementer_records) == 0

    # But should appear in researcher's namespace
    researcher_records = await researcher_mgr.list_all()
    assert len(researcher_records) == 1


# ---------------------------------------------------------------------------
# 4. SharedMemoryManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_to_team_succeeds(mock_store: Any) -> None:
    pmm = PersistentMemoryManager(mock_store)
    smm = SharedMemoryManager(pmm)

    record = _make_record(
        title="Architecture notes",
        memory_type=MemoryType.PROJECT,
        scope=MemoryScope.PROJECT,
        body="We use hexagonal architecture.",
    )
    published = await smm.publish_to_team(record)

    assert published.scope == MemoryScope.TEAM
    assert published.title == "Architecture notes"


@pytest.mark.asyncio
async def test_publish_rejects_secrets(mock_store: Any) -> None:
    pmm = PersistentMemoryManager(mock_store)
    smm = SharedMemoryManager(pmm)

    record = _make_record(
        title="Config",
        memory_type=MemoryType.PROJECT,
        body="api_key=sk-abc123secretkey1234567890abcdef",
    )

    with pytest.raises(SecretDetectedError):
        await smm.publish_to_team(record)


@pytest.mark.asyncio
async def test_publish_rejects_non_shareable_type(mock_store: Any) -> None:
    pmm = PersistentMemoryManager(mock_store)
    smm = SharedMemoryManager(pmm)

    record = _make_record(
        title="User pref", memory_type=MemoryType.USER, body="I like dark mode"
    )

    with pytest.raises(ValueError, match="not shareable"):
        await smm.publish_to_team(record)


def test_scan_for_secrets_detects_patterns() -> None:
    pmm_mock = MagicMock()
    smm = SharedMemoryManager(pmm_mock)

    assert smm.scan_for_secrets("Bearer eyJhbGciOiJIUzI1NiJ9.test")
    assert smm.scan_for_secrets("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    assert smm.scan_for_secrets("AKIAIOSFODNN7EXAMPLE")
    assert smm.scan_for_secrets("api_key=sk-abc123secretkey1234567890abcdef")
    # Clean text should pass
    assert smm.scan_for_secrets("This is a normal project note.") == []


# ---------------------------------------------------------------------------
# 5. MemoryConsolidator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_skips_single_record(mock_store: Any) -> None:
    pmm = PersistentMemoryManager(mock_store)
    record = _make_record(title="Only one")
    record.scope = MemoryScope.USER
    await pmm.create(record)

    mock_llm = AsyncMock()
    consolidator = MemoryConsolidator(pmm, mock_llm)
    result = await consolidator.consolidate(scope=MemoryScope.USER)

    assert result.total_actions == 0
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_consolidate_deletes_stale(mock_store: Any) -> None:
    pmm = PersistentMemoryManager(mock_store)

    r1 = _make_record(title="Stale info")
    r1.scope = MemoryScope.USER
    await pmm.create(r1)

    r2 = _make_record(title="Good info")
    r2.scope = MemoryScope.USER
    await pmm.create(r2)

    # LLM returns a delete action for r1 and keep for r2
    llm_response = MagicMock()
    llm_response.content = json.dumps(
        [
            {"action": "delete", "id": r1.id, "reason": "stale"},
            {"action": "keep", "id": r2.id},
        ]
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = llm_response

    consolidator = MemoryConsolidator(pmm, mock_llm)
    result = await consolidator.consolidate(scope=MemoryScope.USER)

    assert result.deleted == 1
    assert result.kept == 1

    # Verify r1 was actually deleted
    deleted_record = await pmm.get(r1.id, MemoryScope.USER)
    assert deleted_record is None

    # r2 should still exist
    kept_record = await pmm.get(r2.id, MemoryScope.USER)
    assert kept_record is not None


# ---------------------------------------------------------------------------
# 6. PluginRegistry
# ---------------------------------------------------------------------------


def test_register_and_collect_tools() -> None:
    registry = PluginRegistry()
    tool_cap = _make_cap("my_tool", "my_tool", "A plugin tool")
    contrib = PluginContribution("plugin_a", tools=[tool_cap])
    registry.register(contrib)

    tools = registry.collect_tools()
    assert len(tools) == 1
    assert tools[0].name == "my_tool"


def test_collect_sections() -> None:
    registry = PluginRegistry()
    section = PromptSection(
        id="plugin_section",
        content="Plugin instructions",
        stability=SectionStability.STABLE,
        priority=50,
    )
    contrib = PluginContribution("plugin_a", sections=[section])
    registry.register(contrib)

    sections = registry.collect_sections()
    assert len(sections) == 1
    assert sections[0].id == "plugin_section"


def test_multiple_plugins() -> None:
    registry = PluginRegistry()
    cap_a = _make_cap("tool_a", "tool_a")
    cap_b = _make_cap("tool_b", "tool_b")
    registry.register(PluginContribution("plugin_a", tools=[cap_a]))
    registry.register(PluginContribution("plugin_b", tools=[cap_b]))

    tools = registry.collect_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# 7. CoordinatorMode
# ---------------------------------------------------------------------------


def test_coordinator_sections_have_condition() -> None:
    for section in COORDINATOR_SECTIONS:
        assert section.condition == "coordinator", (
            f"Section '{section.id}' missing condition='coordinator'"
        )


def test_get_conditions() -> None:
    conditions = CoordinatorMode.get_conditions()
    assert "coordinator" in conditions


# ---------------------------------------------------------------------------
# 8. adapt_mcp_tool / adapt_mcp_tools (R1-011)
# ---------------------------------------------------------------------------


def test_mcp_adapt_tool() -> None:
    cap = adapt_mcp_tool(
        "test_server",
        name="read_doc",
        description="Read a document",
        fn=lambda: "content",
        risk=ToolRisk.READ_ONLY,
    )

    assert cap.id == "mcp_test_server_read_doc"
    assert cap.name == "read_doc"
    assert "mcp:test_server" in cap.tags
    assert cap.prompt_guidance is not None
    assert "test_server" in cap.prompt_guidance


def test_mcp_adapt_many() -> None:
    tool_defs = [
        {"name": "get_user", "description": "Get user info", "fn": lambda: None},
        {
            "name": "update_user",
            "description": "Update user",
            "fn": lambda: None,
            "risk": "mutating",
            "tags": ["admin"],
        },
    ]
    caps = adapt_mcp_tools("api_server", tool_defs)

    assert len(caps) == 2
    assert caps[0].id == "mcp_api_server_get_user"
    assert caps[0].risk == ToolRisk.READ_ONLY
    assert caps[1].id == "mcp_api_server_update_user"
    assert caps[1].risk == ToolRisk.MUTATING
    assert "admin" in caps[1].tags


# ---------------------------------------------------------------------------
# 9. Activation Prompts (R1-014)
# ---------------------------------------------------------------------------


def test_activation_sections_are_conditional() -> None:
    for section in ACTIVATION_SECTIONS:
        assert section.stability == SectionStability.CONDITIONAL, (
            f"Section '{section.id}' should be CONDITIONAL"
        )


def test_activation_sections_have_conditions() -> None:
    expected_conditions = {"deferred_tools", "skills", "extensions", "async_tasks"}
    actual_conditions = {s.condition for s in ACTIVATION_SECTIONS}
    assert actual_conditions == expected_conditions


def test_activation_sections_included_when_condition_active() -> None:
    from langgraph_kit.core.prompt_assembly.sections import SectionRegistry

    registry = SectionRegistry()
    registry.register_many(ACTIVATION_SECTIONS)

    # Without conditions, no activation sections
    active_none = registry.get_active(conditions=set())
    assert len(active_none) == 0

    # With "deferred_tools" condition, only that section
    active_deferred = registry.get_active(conditions={"deferred_tools"})
    assert len(active_deferred) == 1
    assert active_deferred[0].id == "deferred_tools_awareness"

    # With all conditions, all sections
    active_all = registry.get_active(
        conditions={"deferred_tools", "skills", "extensions"}
    )
    assert len(active_all) == 3
