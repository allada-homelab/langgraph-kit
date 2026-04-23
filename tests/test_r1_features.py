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
    build_call_deferred_tool,
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


@pytest.mark.asyncio
async def test_large_result_without_model_copy_falls_back_to_fresh_toolmessage(
    mock_store: Any,
) -> None:
    """If the result object has no ``model_copy`` the middleware must still
    replace the inline content — not silently return the huge original.

    Regression test for a bug where the ``hasattr`` guard fell through to
    ``return result``: the store write succeeded but the unmodified large
    content kept flowing to the agent, wasting context and orphaning the
    persisted copy.
    """
    middleware = ResultPersistenceMiddleware(persist_threshold=4000, preview_length=50)

    request = MagicMock()
    request.tool_call = {"name": "read_file", "id": "call_no_copy"}
    request.runtime.store = mock_store

    # A bare object with .content but no model_copy (e.g. a plain dataclass
    # a consumer might use instead of ToolMessage).
    class Bare:
        content = "y" * 6000
        id = "orig-id"

    handler = AsyncMock(return_value=Bare())
    result = await middleware.awrap_tool_call(request, handler)

    # Fallback constructed a fresh ToolMessage with the replacement.
    assert "[Full result persisted" in result.content
    assert len(result.content) < 500, (
        "Replacement content must be SMALL — the whole point of persistence"
    )
    # Store still holds the canonical copy.
    stored = mock_store._data.get(("tool_results",))
    assert stored is not None
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_persist_failure_keeps_inline_and_does_not_orphan_store(
    mock_store: Any,
) -> None:
    """If the store write itself fails the original result must flow through.

    The agent can only retrieve what was actually persisted — returning a
    preview that points at a missing store entry would be worse than
    leaving the large content in-message.
    """
    middleware = ResultPersistenceMiddleware(persist_threshold=10)

    # Patch the store to raise on aput.
    async def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("store down")

    mock_store.aput = _boom  # type: ignore[attr-defined]

    request = MagicMock()
    request.tool_call = {"name": "t", "id": "c"}
    request.runtime.store = mock_store

    result_msg = MagicMock()
    result_msg.content = "large content that should persist"
    result_msg.model_copy = lambda *, update: MagicMock(content=update["content"])

    handler = AsyncMock(return_value=result_msg)
    result = await middleware.awrap_tool_call(request, handler)

    # Original content preserved — no dangling preview.
    assert result is result_msg
    assert result.content == "large content that should persist"


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


def test_deferred_registry_bool_and_len_reflect_content() -> None:
    """Builder code relies on ``bool(registry)`` to gate the deferred_tools condition."""
    registry = DeferredToolRegistry()
    assert len(registry) == 0
    assert not registry  # empty is falsy

    registry.register(_make_cap("fs", "file_search"))
    assert len(registry) == 1
    assert registry  # populated is truthy


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


@pytest.mark.asyncio
async def test_tool_search_advertises_dispatcher() -> None:
    """The search output MUST tell the agent how to invoke a discovered tool.

    Regression test for a documented design flaw: earlier versions said
    "call it by name — it will be activated automatically", but deferred
    tools aren't bound to the LLM, so name-calling never reaches them.
    Output must advertise ``call_deferred_tool`` instead, with the id
    field + argument dict shape, so the model can actually reach the tool.
    """
    registry = DeferredToolRegistry()
    registry.register(
        _make_cap("fs", "file_search", "Search files", tags=["filesystem"])
    )

    output = await build_tool_search(registry)("file")
    assert "call_deferred_tool" in output
    # The id must be rendered distinctly from the display name so the LLM
    # doesn't conflate the two when building its dispatcher call.
    assert "id: `fs`" in output
    # Signature hint so the LLM knows the argument shape.
    assert "Signature:" in output


@pytest.mark.asyncio
async def test_call_deferred_tool_invokes_registered_callable() -> None:
    """End-to-end: register a deferred tool, then call it via the dispatcher.

    This is the happy path that the whole deferred-tool feature hinges on.
    Before the fix there was no code path from tool_search to actual
    invocation — this test would have been impossible to write.
    """
    registry = DeferredToolRegistry()

    async def deploy(environment: str, dry_run: bool = False) -> str:
        return f"deploy: env={environment} dry_run={dry_run}"

    registry.register(
        ToolCapability(
            id="deploy_tool",
            name="deploy",
            description="Deploy to an environment",
            fn=deploy,
            tags=["ops"],
            risk=ToolRisk.MUTATING,
        )
    )

    dispatcher = build_call_deferred_tool(registry)
    result = await dispatcher(
        tool_id="deploy_tool", arguments={"environment": "staging", "dry_run": True}
    )
    assert result == "deploy: env=staging dry_run=True"


@pytest.mark.asyncio
async def test_call_deferred_tool_supports_sync_callables() -> None:
    """Dispatcher must work whether the registered tool is sync or async.

    Real-world deferred catalogs include both — MCP tools tend to be
    async, but plain Python helpers registered by user code are often
    synchronous. Both must round-trip through the dispatcher.
    """
    registry = DeferredToolRegistry()

    def compute(a: int, b: int) -> int:
        return a + b

    registry.register(
        ToolCapability(
            id="add",
            name="compute",
            description="Add two integers",
            fn=compute,
            risk=ToolRisk.READ_ONLY,
        )
    )

    dispatcher = build_call_deferred_tool(registry)
    result = await dispatcher(tool_id="add", arguments={"a": 2, "b": 3})
    # Non-string returns are JSON-serialized so the model gets a stable shape.
    assert result == "5"


@pytest.mark.asyncio
async def test_call_deferred_tool_unknown_id_returns_error_string() -> None:
    """Unknown tool ids must not raise — return an error message the LLM can read.

    Tool functions that raise in-flight break the agent loop. Returning
    a string keeps the model in control and gives it enough context to
    re-query ``tool_search`` for the right id.
    """
    registry = DeferredToolRegistry()
    registry.register(_make_cap("known", "known_tool"))

    dispatcher = build_call_deferred_tool(registry)
    result = await dispatcher(tool_id="unknown", arguments={})
    assert "not found" in result
    # Should surface the available ids so the model can recover.
    assert "known" in result


@pytest.mark.asyncio
async def test_call_deferred_tool_parses_json_string_arguments() -> None:
    """Some LLMs emit ``arguments`` as a JSON string instead of a dict.

    Rejecting the call there would train the model to avoid the
    dispatcher entirely, so the common case of a stringified dict is
    decoded transparently. Malformed strings still fail with a useful
    message.
    """
    registry = DeferredToolRegistry()

    async def echo(msg: str) -> str:
        return f"echo: {msg}"

    registry.register(
        ToolCapability(
            id="echo",
            name="echo",
            description="Echo a message",
            fn=echo,
            risk=ToolRisk.READ_ONLY,
        )
    )

    dispatcher = build_call_deferred_tool(registry)
    result = await dispatcher(tool_id="echo", arguments='{"msg": "hi"}')  # type: ignore[arg-type]
    assert result == "echo: hi"


@pytest.mark.asyncio
async def test_call_deferred_tool_wrong_args_returns_error_string() -> None:
    """Argument shape mismatches must come back as an LLM-readable error."""
    registry = DeferredToolRegistry()

    async def needs_name(name: str) -> str:
        return name

    registry.register(
        ToolCapability(
            id="greet",
            name="greet",
            description="Greet by name",
            fn=needs_name,
            risk=ToolRisk.READ_ONLY,
        )
    )

    dispatcher = build_call_deferred_tool(registry)
    # Wrong kwarg name — the underlying TypeError would normally abort
    # the tool call; the dispatcher must return a string instead.
    result = await dispatcher(tool_id="greet", arguments={"wrong_kw": "x"})
    assert "Error calling 'greet'" in result


def test_register_search_tool_registers_both_search_and_dispatcher() -> None:
    """``register_search_tool`` must put BOTH tools in the active registry.

    Before the fix it only registered ``tool_search`` — the dispatcher
    was missing entirely, so any tool discovered via search was
    effectively unreachable.
    """
    from langgraph_kit.core.graph_builder.tools import register_search_tool
    from langgraph_kit.core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    deferred = register_search_tool(registry)

    ids = {cap.id for cap in registry.list_all()}
    assert "tool_search" in ids
    assert "call_deferred_tool" in ids
    # Returned deferred registry is the one both tools are bound to.
    assert isinstance(deferred, DeferredToolRegistry)


def test_register_search_tool_accepts_existing_deferred_registry() -> None:
    """Callers can pass an existing ``DeferredToolRegistry`` to bind both tools against.

    The builder creates the registry early so plugin/configure callbacks
    can populate it, then registers the search + dispatcher tools at the
    end only when the catalog is non-empty. That "register against an
    existing registry" path must be supported so tools discovered via
    ``tool_search`` dispatch back to the same registry the caller populated.
    """
    from langgraph_kit.core.graph_builder.tools import register_search_tool
    from langgraph_kit.core.tools.registry import ToolRegistry

    pre_existing = DeferredToolRegistry()
    pre_existing.register(
        ToolCapability(
            id="pre",
            name="pre",
            description="a tool registered before search tools were bound",
            fn=lambda: "pre",
            risk=ToolRisk.READ_ONLY,
        )
    )

    registry = ToolRegistry()
    returned = register_search_tool(registry, pre_existing)

    assert returned is pre_existing, (
        "register_search_tool must reuse the registry it was handed"
    )
    ids = {cap.id for cap in registry.list_all()}
    assert "tool_search" in ids
    assert "call_deferred_tool" in ids


@pytest.mark.asyncio
async def test_call_deferred_tool_accepts_json_string_through_langchain_tool() -> None:
    """LangChain-wrapped ``call_deferred_tool`` must accept stringified JSON arguments.

    Some LLMs (notably Qwen variants) emit the ``arguments`` field of a
    tool call as a JSON string instead of a dict. The in-body coercion
    only runs if Pydantic lets the value through — so the function's
    type annotation must admit both dict and str, otherwise LangChain's
    ``StructuredTool`` rejects the call before the coercion fires and
    the agent spins retrying the same shape.
    """
    from langchain_core.tools import tool as _lc_tool

    registry = DeferredToolRegistry()

    async def echo(msg: str) -> str:
        return f"echo: {msg}"

    registry.register(
        ToolCapability(
            id="echo",
            name="echo",
            description="Echo a message",
            fn=echo,
            risk=ToolRisk.READ_ONLY,
        )
    )

    wrapped = _lc_tool(build_call_deferred_tool(registry))

    # Dict payload — baseline.
    dict_result = await wrapped.ainvoke({"tool_id": "echo", "arguments": {"msg": "hi"}})
    assert dict_result == "echo: hi"

    # JSON-string payload — must round-trip without a ValidationError.
    str_result = await wrapped.ainvoke(
        {"tool_id": "echo", "arguments": '{"msg": "via string"}'}
    )
    assert str_result == "echo: via string"


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
async def test_consolidate_skips_merge_with_invalid_type(mock_store: Any) -> None:
    """An invalid ``type`` on a merge action must not crash the whole pass.

    Companion regression to the extractor enum bug: consolidation used
    ``MemoryType(merged_data.get("type", "user"))`` with no guard, so an LLM
    invention like ``"type": "assistant"`` would raise ValueError on merge.
    With coerce_memory_type, the bad merge is skipped (recorded as an error)
    and the source records are preserved instead of being deleted with no
    replacement.
    """
    pmm = PersistentMemoryManager(mock_store)

    r1 = _make_record(title="A")
    r1.scope = MemoryScope.USER
    await pmm.create(r1)
    r2 = _make_record(title="B")
    r2.scope = MemoryScope.USER
    await pmm.create(r2)

    # Bad merge: type "assistant" is not a MemoryType.
    # Sibling action (keep on r1) must still be counted.
    llm_response = MagicMock()
    llm_response.content = json.dumps(
        [
            {
                "action": "merge",
                "source_ids": [r1.id, r2.id],
                "merged": {
                    "title": "Merged",
                    "type": "assistant",
                    "summary": "s",
                    "body": "b",
                },
            },
            {"action": "keep", "id": r1.id},
        ]
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = llm_response

    consolidator = MemoryConsolidator(pmm, mock_llm)
    result = await consolidator.consolidate(scope=MemoryScope.USER)

    # Merge was skipped, keep still counted.
    assert result.merged == 0
    assert result.kept == 1
    # Source records preserved — they were NOT deleted by a partially-applied
    # merge.
    assert await pmm.get(r1.id, MemoryScope.USER) is not None
    assert await pmm.get(r2.id, MemoryScope.USER) is not None


@pytest.mark.asyncio
async def test_consolidate_tags_llm_call_as_internal(mock_store: Any) -> None:
    """MemoryConsolidator.ainvoke must carry INTERNAL_TAG + consolidation tag.

    Same leak class as extraction — without tagging, the consolidator's
    JSON-action stream would leak into any user-facing transcript that
    happened to be running (rare in practice, but the filter is cheap).
    """
    from langgraph_kit.core.internal_tags import (
        INTERNAL_TAG,
        MEMORY_CONSOLIDATION_TAG,
    )

    pmm = PersistentMemoryManager(mock_store)
    for i in range(2):
        rec = _make_record(title=f"Rec {i}")
        rec.scope = MemoryScope.USER
        await pmm.create(rec)

    llm_response = MagicMock()
    llm_response.content = "[]"
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = llm_response

    consolidator = MemoryConsolidator(pmm, mock_llm)
    await consolidator.consolidate(scope=MemoryScope.USER)

    mock_llm.ainvoke.assert_awaited_once()
    config = mock_llm.ainvoke.call_args.kwargs.get("config")
    assert config is not None
    tags = config.get("tags", [])
    assert INTERNAL_TAG in tags
    assert MEMORY_CONSOLIDATION_TAG in tags
    assert config.get("run_name") == "memory_consolidation"


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
