"""Coverage fill — ``SharedMemoryManager`` publish + sync flows."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.memory.models import MemoryRecord, MemoryScope, MemoryType
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.memory.shared import (
    SecretDetectedError,
    SharedMemoryManager,
)


def _rec(**overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "title": "k8s deploy target",
        "type": MemoryType.PROJECT,
        "scope": MemoryScope.PROJECT,
        "summary": "short",
        "body": "Deploy to production K8s cluster via helm.",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


@pytest.fixture
def mgr(mock_store: Any) -> SharedMemoryManager:
    return SharedMemoryManager(PersistentMemoryManager(mock_store))


# ---------------------------------------------------------------------------
# scan_for_secrets
# ---------------------------------------------------------------------------


def test_scan_for_secrets_matches_openai_key(mgr: SharedMemoryManager) -> None:
    hits = mgr.scan_for_secrets(
        "Use this key: sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
    )
    assert hits


def test_scan_for_secrets_matches_github_pat(mgr: SharedMemoryManager) -> None:
    assert mgr.scan_for_secrets("token=ghp_" + "a" * 40)


def test_scan_for_secrets_matches_aws_key(mgr: SharedMemoryManager) -> None:
    assert mgr.scan_for_secrets("export AKIAIOSFODNN7EXAMPLE")


def test_scan_for_secrets_matches_private_key_header(
    mgr: SharedMemoryManager,
) -> None:
    assert mgr.scan_for_secrets("-----BEGIN RSA PRIVATE KEY-----\nfake")


def test_scan_for_secrets_clean_text(mgr: SharedMemoryManager) -> None:
    assert mgr.scan_for_secrets("Deploy to prod cluster via helm.") == []


# ---------------------------------------------------------------------------
# publish_to_team
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_project_memory_to_team(
    mgr: SharedMemoryManager,
) -> None:
    rec = _rec()
    published = await mgr.publish_to_team(rec)
    assert published.scope == MemoryScope.TEAM
    assert published.source is not None
    assert "published_from:" in published.source


@pytest.mark.asyncio
async def test_publish_rejects_non_shareable_type_by_default(
    mgr: SharedMemoryManager,
) -> None:
    """FEEDBACK / USER types require ``allow_all_types=True`` to share."""
    rec = _rec(type=MemoryType.FEEDBACK)
    with pytest.raises(ValueError, match="not shareable"):
        await mgr.publish_to_team(rec)


@pytest.mark.asyncio
async def test_publish_non_shareable_with_override_succeeds(
    mgr: SharedMemoryManager,
) -> None:
    rec = _rec(type=MemoryType.FEEDBACK)
    published = await mgr.publish_to_team(rec, allow_all_types=True)
    assert published.scope == MemoryScope.TEAM


@pytest.mark.asyncio
async def test_publish_rejects_secret_bodied_memory(
    mgr: SharedMemoryManager,
) -> None:
    rec = _rec(body="credentials: api_key=pk_live_EXAMPLE-12345xyzXYZ")
    with pytest.raises(SecretDetectedError):
        await mgr.publish_to_team(rec)


# ---------------------------------------------------------------------------
# sync_from_team + list_team_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_from_team_copies_each_missing_record(
    mgr: SharedMemoryManager,
    mock_store: Any,
) -> None:
    pm = PersistentMemoryManager(mock_store)
    # Seed team scope directly.
    await pm.create(_rec(title="team-A", scope=MemoryScope.TEAM))
    await pm.create(_rec(title="team-B", scope=MemoryScope.TEAM))

    synced = await mgr.sync_from_team(target_scope=MemoryScope.PROJECT)
    assert {r.title for r in synced} == {"team-A", "team-B"}
    # Records now live in PROJECT scope.
    ns = ("memory", MemoryScope.PROJECT.value, MemoryType.PROJECT.value)
    assert len(mock_store._data.get(ns, {})) == 2


@pytest.mark.asyncio
async def test_sync_from_team_dedupes_by_title_and_type(
    mgr: SharedMemoryManager,
    mock_store: Any,
) -> None:
    pm = PersistentMemoryManager(mock_store)
    # Pre-existing record in PROJECT scope with matching (title, type).
    await pm.create(_rec(title="dup", scope=MemoryScope.PROJECT))
    # Team record with same title+type → should be skipped.
    await pm.create(_rec(title="dup", scope=MemoryScope.TEAM))

    synced = await mgr.sync_from_team(target_scope=MemoryScope.PROJECT)
    assert synced == []


@pytest.mark.asyncio
async def test_list_team_memories_filters_by_type(
    mgr: SharedMemoryManager,
    mock_store: Any,
) -> None:
    pm = PersistentMemoryManager(mock_store)
    await pm.create(
        _rec(title="team-proj", type=MemoryType.PROJECT, scope=MemoryScope.TEAM)
    )
    await pm.create(
        _rec(title="team-ref", type=MemoryType.REFERENCE, scope=MemoryScope.TEAM)
    )
    refs = await mgr.list_team_memories(memory_type=MemoryType.REFERENCE)
    assert [r.title for r in refs] == ["team-ref"]
