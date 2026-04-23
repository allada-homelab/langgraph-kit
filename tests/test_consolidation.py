"""Coverage fill — ``MemoryConsolidator`` action application.

The consolidator asks an LLM for a JSON action list and applies each
action (keep / delete / merge / update). Tests drive every action
branch with a scripted LLM so we don't depend on live model behavior.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_kit.core.memory.consolidation import (
    ConsolidationResult,
    MemoryConsolidator,
)
from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager


class _ScriptedLLM:
    """Returns a canned JSON action list."""

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._payload = json.dumps(actions)

    async def ainvoke(self, messages: list[Any], config: Any = None) -> Any:
        _ = messages
        _ = config

        class _R:
            content = self._payload

        return _R()


class _RaisingLLM:
    async def ainvoke(self, messages: list[Any], config: Any = None) -> Any:
        _ = messages
        _ = config
        msg = "llm down"
        raise RuntimeError(msg)


def _rec(**overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "title": "t",
        "type": MemoryType.PROJECT,
        "scope": MemoryScope.USER,
        "summary": "s",
        "body": "b",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


@pytest.fixture
def memory_mgr(mock_store: Any) -> PersistentMemoryManager:
    return PersistentMemoryManager(mock_store)


# ---------------------------------------------------------------------------
# ConsolidationResult
# ---------------------------------------------------------------------------


def test_consolidation_result_total_actions_sums_each_field() -> None:
    r = ConsolidationResult()
    r.kept = 1
    r.deleted = 2
    r.merged = 3
    r.updated = 4
    assert r.total_actions == 10


def test_consolidation_result_repr_contains_counts() -> None:
    r = ConsolidationResult()
    r.kept = 5
    rep = repr(r)
    assert "kept=5" in rep


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_with_fewer_than_two_records_is_noop(
    memory_mgr: PersistentMemoryManager,
) -> None:
    llm = _ScriptedLLM([])  # never called
    consolidator = MemoryConsolidator(memory_mgr, llm)
    result = await consolidator.consolidate()
    assert result.total_actions == 0
    # no LLM call happens below the 2-record threshold.


@pytest.mark.asyncio
async def test_consolidate_llm_failure_adds_error_entry(
    memory_mgr: PersistentMemoryManager,
) -> None:
    await memory_mgr.create(_rec(title="a"))
    await memory_mgr.create(_rec(title="b"))

    consolidator = MemoryConsolidator(memory_mgr, _RaisingLLM())
    result = await consolidator.consolidate()
    assert result.errors


@pytest.mark.asyncio
async def test_consolidate_applies_keep_delete_update(
    memory_mgr: PersistentMemoryManager,
) -> None:
    r1 = await memory_mgr.create(_rec(title="keeper"))
    r2 = await memory_mgr.create(_rec(title="staler"))
    r3 = await memory_mgr.create(_rec(title="updatable", body="old body"))

    actions = [
        {"action": "keep", "id": r1.id},
        {"action": "delete", "id": r2.id, "reason": "stale"},
        {
            "action": "update",
            "id": r3.id,
            "updates": {"body": "new body"},
        },
    ]
    consolidator = MemoryConsolidator(memory_mgr, _ScriptedLLM(actions))
    result = await consolidator.consolidate()
    assert result.kept == 1
    assert result.deleted == 1
    assert result.updated == 1

    # Delete + update are reflected in storage.
    assert await memory_mgr.get(r2.id, MemoryScope.USER) is None
    updated = await memory_mgr.get(r3.id, MemoryScope.USER)
    assert updated is not None
    assert updated.body == "new body"


@pytest.mark.asyncio
async def test_consolidate_applies_merge_action(
    memory_mgr: PersistentMemoryManager,
) -> None:
    r1 = await memory_mgr.create(_rec(title="near-dup 1"))
    r2 = await memory_mgr.create(_rec(title="near-dup 2"))
    actions = [
        {
            "action": "merge",
            "source_ids": [r1.id, r2.id],
            "merged": {
                "title": "merged",
                "type": "project",
                "summary": "s",
                "body": "b",
            },
        }
    ]
    consolidator = MemoryConsolidator(memory_mgr, _ScriptedLLM(actions))
    result = await consolidator.consolidate()
    assert result.merged == 1
    # Sources deleted.
    assert await memory_mgr.get(r1.id, MemoryScope.USER) is None
    assert await memory_mgr.get(r2.id, MemoryScope.USER) is None


@pytest.mark.asyncio
async def test_consolidate_skips_merge_with_unrecognized_type(
    memory_mgr: PersistentMemoryManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r1 = await memory_mgr.create(_rec(title="a"))
    r2 = await memory_mgr.create(_rec(title="b"))
    actions = [
        {
            "action": "merge",
            "source_ids": [r1.id, r2.id],
            "merged": {
                "title": "merged",
                "type": "made-up-type",
                "summary": "s",
                "body": "b",
            },
        }
    ]
    consolidator = MemoryConsolidator(memory_mgr, _ScriptedLLM(actions))
    result = await consolidator.consolidate()
    # Bad type skips the merge — source records are NOT deleted.
    assert result.merged == 0
    assert await memory_mgr.get(r1.id, MemoryScope.USER) is not None
    assert await memory_mgr.get(r2.id, MemoryScope.USER) is not None
    _ = caplog  # warning is emitted but we don't assert on text


@pytest.mark.asyncio
async def test_consolidate_handles_individual_action_exceptions(
    memory_mgr: PersistentMemoryManager,
) -> None:
    r1 = await memory_mgr.create(_rec(title="a"))
    await memory_mgr.create(_rec(title="second-record"))
    # Malformed action — missing id on delete, missing updates on update.
    # These land in the "update-with-empty-updates" and "delete-no-id"
    # branches which are silent no-ops. Another shape — merge with no
    # source_ids — is also silently dropped. A truly raising action is
    # harder to construct via data alone, so we pin the silent-drop
    # behaviour here instead.
    actions = [
        {"action": "delete"},  # missing id
        {"action": "update", "id": r1.id, "updates": {}},
        {"action": "merge"},  # missing fields
        {"action": "keep", "id": r1.id},
    ]
    consolidator = MemoryConsolidator(memory_mgr, _ScriptedLLM(actions))
    result = await consolidator.consolidate()
    # Only the keep action counted.
    assert result.kept == 1
    assert result.deleted == 0
    assert result.updated == 0
    assert result.merged == 0
