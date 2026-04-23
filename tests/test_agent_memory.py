"""Coverage fill — ``AgentMemoryManager`` worker-scoped CRUD."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.memory.agent_memory import AgentMemoryManager
from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)


def _record(**overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "title": "agent-memory",
        "type": MemoryType.PROJECT,
        "scope": MemoryScope.USER,
        "summary": "s",
        "body": "b",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


@pytest.fixture
def manager(mock_store: Any) -> AgentMemoryManager:
    return AgentMemoryManager(mock_store, agent_name="researcher")


@pytest.mark.asyncio
async def test_create_and_get_round_trip(manager: AgentMemoryManager) -> None:
    rec = _record()
    created = await manager.create(rec)
    assert created.id == rec.id

    fetched = await manager.get(rec.id, memory_type=rec.type)
    assert fetched is not None
    assert fetched.title == "agent-memory"


@pytest.mark.asyncio
async def test_get_without_type_probes_every_namespace(
    manager: AgentMemoryManager,
) -> None:
    rec = _record(type=MemoryType.REFERENCE)
    await manager.create(rec)

    fetched = await manager.get(rec.id, memory_type=None)
    assert fetched is not None
    assert fetched.type == MemoryType.REFERENCE


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_id(manager: AgentMemoryManager) -> None:
    assert await manager.get("no-such-id") is None


@pytest.mark.asyncio
async def test_list_all_filtered_by_type(manager: AgentMemoryManager) -> None:
    await manager.create(_record(title="proj1", type=MemoryType.PROJECT))
    await manager.create(_record(title="ref1", type=MemoryType.REFERENCE))
    project_records = await manager.list_all(memory_type=MemoryType.PROJECT)
    titles = {r.title for r in project_records}
    assert titles == {"proj1"}


@pytest.mark.asyncio
async def test_list_all_unfiltered_collects_every_type(
    manager: AgentMemoryManager,
) -> None:
    await manager.create(_record(title="p", type=MemoryType.PROJECT))
    await manager.create(_record(title="r", type=MemoryType.REFERENCE))
    everyone = await manager.list_all()
    assert {r.title for r in everyone} == {"p", "r"}


@pytest.mark.asyncio
async def test_update_mutates_in_place(manager: AgentMemoryManager) -> None:
    rec = _record(body="before")
    await manager.create(rec)
    updated = await manager.update(rec.id, {"body": "after"}, memory_type=rec.type)
    assert updated is not None
    assert updated.body == "after"


@pytest.mark.asyncio
async def test_update_unknown_id_returns_none(
    manager: AgentMemoryManager,
) -> None:
    assert await manager.update("ghost", {"body": "x"}) is None


@pytest.mark.asyncio
async def test_update_relocates_record_on_type_change(
    manager: AgentMemoryManager, mock_store: Any
) -> None:
    rec = _record(type=MemoryType.PROJECT)
    await manager.create(rec)
    old_ns = ("memory", "agent", "researcher", MemoryType.PROJECT.value)
    assert rec.id in mock_store._data.get(old_ns, {})

    updated = await manager.update(
        rec.id, {"type": MemoryType.REFERENCE.value}, memory_type=rec.type
    )
    assert updated is not None
    assert updated.type == MemoryType.REFERENCE

    new_ns = ("memory", "agent", "researcher", MemoryType.REFERENCE.value)
    assert rec.id in mock_store._data.get(new_ns, {})
    assert rec.id not in mock_store._data.get(old_ns, {})


@pytest.mark.asyncio
async def test_delete_removes_record(manager: AgentMemoryManager) -> None:
    rec = _record()
    await manager.create(rec)
    assert await manager.delete(rec.id, memory_type=rec.type) is True
    assert await manager.get(rec.id, memory_type=rec.type) is None


@pytest.mark.asyncio
async def test_delete_unknown_id_returns_false(
    manager: AgentMemoryManager,
) -> None:
    assert await manager.delete("no-such-id") is False


@pytest.mark.asyncio
async def test_snapshot_from_copies_each_record_into_agent_namespace(
    manager: AgentMemoryManager,
) -> None:
    sources = [
        _record(title="a", type=MemoryType.PROJECT),
        _record(title="b", type=MemoryType.REFERENCE),
    ]
    count = await manager.snapshot_from(sources)
    assert count == 2

    listed = await manager.list_all()
    assert {r.title for r in listed} == {"a", "b"}
    # Source attribution: every snapshotted record tags the originating id.
    assert all((r.source or "").startswith("snapshot_from:") for r in listed)
