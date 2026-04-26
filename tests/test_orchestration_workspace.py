"""Tests for ``langgraph_kit.core.orchestration.workspace`` (issue #20 v1)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from langgraph_kit.core.orchestration.workspace import (
    AgentWorkspace,
    WorkspaceConflict,
)
from tests.conftest import MockStore


class _TaskBoard(BaseModel):
    """Toy schema for the workspace under test — task lists in two buckets."""

    todo: list[str] = []
    done: list[str] = []


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


# ---------------------------------------------------------------------------
# Read / write happy paths.
# ---------------------------------------------------------------------------


class TestAgentWorkspaceBasics:
    @pytest.mark.asyncio
    async def test_aget_empty_returns_none(self, mock_store: MockStore) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        assert await workspace.aget() is None
        assert await workspace.aget_with_revision() is None

    @pytest.mark.asyncio
    async def test_aput_then_aget_round_trip(self, mock_store: MockStore) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        revision = await workspace.aput(_TaskBoard(todo=["task-a"]))
        assert revision == 1
        loaded = await workspace.aget()
        assert loaded is not None
        assert loaded.todo == ["task-a"]
        assert loaded.done == []

    @pytest.mark.asyncio
    async def test_aput_increments_revision_each_call(
        self, mock_store: MockStore
    ) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        rev1 = await workspace.aput(_TaskBoard(todo=["a"]))
        rev2 = await workspace.aput(_TaskBoard(todo=["b"]))
        rev3 = await workspace.aput(_TaskBoard(todo=["c"]))
        assert (rev1, rev2, rev3) == (1, 2, 3)

    @pytest.mark.asyncio
    async def test_aget_with_revision_returns_pair(self, mock_store: MockStore) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))
        result = await workspace.aget_with_revision()
        assert result is not None
        doc, rev = result
        assert doc.todo == ["x"]
        assert rev == 1

    @pytest.mark.asyncio
    async def test_revision_is_not_in_user_schema(self, mock_store: MockStore) -> None:
        """``_revision`` lives on the wire format, not the user model."""
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))
        loaded = await workspace.aget()
        assert loaded is not None
        # _TaskBoard doesn't declare ``_revision`` and shouldn't gain it.
        assert not hasattr(loaded, "_revision")

    @pytest.mark.asyncio
    async def test_adelete_removes_workspace(self, mock_store: MockStore) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))
        await workspace.adelete()
        assert await workspace.aget() is None


# ---------------------------------------------------------------------------
# Optimistic concurrency.
# ---------------------------------------------------------------------------


class TestAgentWorkspaceConcurrency:
    @pytest.mark.asyncio
    async def test_apatch_with_matching_revision_succeeds(
        self, mock_store: MockStore
    ) -> None:
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))
        result = await workspace.aget_with_revision()
        assert result is not None
        doc, rev = result
        doc.todo.append("y")
        new_rev = await workspace.apatch(doc, expected_revision=rev)
        assert new_rev == rev + 1
        loaded = await workspace.aget()
        assert loaded is not None
        assert loaded.todo == ["x", "y"]

    @pytest.mark.asyncio
    async def test_apatch_with_stale_revision_raises_conflict(
        self, mock_store: MockStore
    ) -> None:
        """Two readers patch concurrently; the second one's stale revision rejects."""
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))

        # Reader A and B both fetch revision 1.
        result_a = await workspace.aget_with_revision()
        result_b = await workspace.aget_with_revision()
        assert result_a is not None
        assert result_b is not None
        doc_a, rev_a = result_a
        doc_b, rev_b = result_b
        assert rev_a == rev_b == 1

        # A patches first — succeeds (revision -> 2).
        doc_a.todo.append("a-update")
        await workspace.apatch(doc_a, expected_revision=rev_a)

        # B patches with the same expected_revision=1 — raises.
        doc_b.todo.append("b-update")
        with pytest.raises(WorkspaceConflict, match="revision moved"):
            await workspace.apatch(doc_b, expected_revision=rev_b)

    @pytest.mark.asyncio
    async def test_apatch_with_retry_handles_one_collision(
        self, mock_store: MockStore
    ) -> None:
        """Helper retries on stale revision and lands the patch."""
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))

        # Simulate a concurrent writer landing one update before our retry loop.
        result = await workspace.aget_with_revision()
        assert result is not None
        doc_other, rev_other = result
        doc_other.todo.append("other-update")
        await workspace.apatch(doc_other, expected_revision=rev_other)

        # Now apatch_with_retry — the first attempt sees revision 2,
        # mutates from there, lands successfully on attempt 1.
        def mutate(doc: _TaskBoard) -> _TaskBoard:
            doc.done.append(doc.todo[-1])
            return doc

        new_doc, new_rev = await workspace.apatch_with_retry(mutate)
        assert new_rev == 3
        assert new_doc.done == ["other-update"]

    @pytest.mark.asyncio
    async def test_apatch_with_retry_exhausts_budget_on_persistent_conflict(
        self, mock_store: MockStore
    ) -> None:
        """If every retry collides, raise the last conflict.

        Simulated by patching ``apatch`` to always raise — no need to
        race actual writers in a single-threaded asyncio loop, where
        cooperative scheduling makes "concurrent" collisions hard to
        reproduce deterministically.
        """
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        await workspace.aput(_TaskBoard(todo=["x"]))

        attempts = 0

        async def always_conflict(*_: object, **__: object) -> int:
            nonlocal attempts
            attempts += 1
            msg = "simulated stale revision"
            raise WorkspaceConflict(msg)

        workspace.apatch = always_conflict  # type: ignore[method-assign]

        with pytest.raises(WorkspaceConflict, match="simulated stale revision"):
            await workspace.apatch_with_retry(lambda doc: doc, retries=3)
        # Each retry called apatch exactly once.
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_apatch_with_retry_on_missing_workspace_raises(
        self, mock_store: MockStore
    ) -> None:
        """Calling apatch_with_retry before aput is a misuse."""
        workspace = AgentWorkspace(mock_store, "board-1", _TaskBoard)
        with pytest.raises(WorkspaceConflict, match="doesn't exist"):
            await workspace.apatch_with_retry(lambda doc: doc)


# ---------------------------------------------------------------------------
# Cross-workspace isolation.
# ---------------------------------------------------------------------------


class TestAgentWorkspaceIsolation:
    @pytest.mark.asyncio
    async def test_distinct_workspace_ids_dont_share_state(
        self, mock_store: MockStore
    ) -> None:
        ws_a = AgentWorkspace(mock_store, "board-a", _TaskBoard)
        ws_b = AgentWorkspace(mock_store, "board-b", _TaskBoard)
        await ws_a.aput(_TaskBoard(todo=["a"]))
        await ws_b.aput(_TaskBoard(todo=["b"]))
        loaded_a = await ws_a.aget()
        loaded_b = await ws_b.aget()
        assert loaded_a is not None
        assert loaded_b is not None
        assert loaded_a.todo == ["a"]
        assert loaded_b.todo == ["b"]

    @pytest.mark.asyncio
    async def test_workspace_id_property_round_trips(
        self, mock_store: MockStore
    ) -> None:
        ws = AgentWorkspace(mock_store, "board-x", _TaskBoard)
        assert ws.workspace_id == "board-x"
