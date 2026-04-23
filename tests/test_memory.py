"""Tests for memory module: models, persistent manager, and session notebook."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    coerce_memory_type,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager
from langgraph_kit.core.memory.session import (
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MESSAGE_THRESHOLD,
    DEFAULT_TOOL_CALL_THRESHOLD,
    NOTEBOOK_SECTIONS,
    NOTEBOOK_TEMPLATE,
    SessionNotebook,
)

# ---------------------------------------------------------------------------
# Fixtures — MockStore comes from conftest.py
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(mock_store: Any) -> PersistentMemoryManager:
    return PersistentMemoryManager(mock_store)


@pytest.fixture
def notebook(mock_store: Any) -> SessionNotebook:
    return SessionNotebook(mock_store, thread_id="test-thread-1")


def _make_record(**kwargs: object) -> MemoryRecord:
    defaults: dict[str, object] = {
        "title": "Test memory",
        "type": MemoryType.USER,
        "scope": MemoryScope.USER,
        "summary": "A short summary",
        "body": "Detailed body content",
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# MemoryRecord model tests (synchronous)
# ===========================================================================


class TestCoerceMemoryType:
    """Pre-validation helper that replaces raw ``MemoryType(value)`` calls.

    Motivated by a crash where an extractor LLM emitted ``"type": "assistant"``
    (outside the taxonomy) and the resulting ValueError polluted error logs
    with a full traceback.
    """

    def test_valid_string_returns_member(self) -> None:
        for name in ("user", "feedback", "project", "reference"):
            assert coerce_memory_type(name) == MemoryType(name)

    def test_already_enum_is_returned_as_is(self) -> None:
        assert coerce_memory_type(MemoryType.FEEDBACK) is MemoryType.FEEDBACK

    def test_invalid_string_returns_none(self) -> None:
        # Real-world values that the extractor has produced.
        for bad in ("assistant", "system", "note", "", "USER"):
            assert coerce_memory_type(bad) is None, bad

    def test_non_string_returns_none(self) -> None:
        for bad in (None, 42, [], {}, object()):
            assert coerce_memory_type(bad) is None


class TestMemoryRecord:
    def test_memory_record_defaults(self) -> None:
        record = _make_record()
        # id should be a UUID string
        assert record.id
        assert len(record.id) == 36  # UUID4 with dashes
        # timestamps should be set and recent
        now = datetime.now(UTC)
        assert (now - record.created_at).total_seconds() < 5
        assert (now - record.updated_at).total_seconds() < 5

    def test_to_store_value_roundtrip(self) -> None:
        record = _make_record(
            title="Roundtrip test",
            type=MemoryType.FEEDBACK,
            scope=MemoryScope.PROJECT,
            summary="Summary here",
            body="Body here",
            source="test-source",
        )
        serialized = record.to_store_value()
        assert isinstance(serialized, dict)
        assert serialized["title"] == "Roundtrip test"
        assert serialized["source"] == "test-source"

        restored = MemoryRecord.from_store_value(serialized)
        assert restored.id == record.id
        assert restored.title == record.title
        assert restored.type == record.type
        assert restored.scope == record.scope
        assert restored.summary == record.summary
        assert restored.body == record.body
        assert restored.source == record.source
        assert restored.created_at == record.created_at
        assert restored.updated_at == record.updated_at


# ===========================================================================
# PersistentMemoryManager tests (async)
# ===========================================================================


class TestPersistentMemoryManager:
    @pytest.mark.asyncio
    async def test_create_memory(self, manager: PersistentMemoryManager) -> None:
        record = _make_record(title="Created record")
        result = await manager.create(record)
        assert result.id == record.id
        assert result.title == "Created record"
        # Verify it's actually in the store
        fetched = await manager.get(record.id, record.scope)
        assert fetched is not None
        assert fetched.title == "Created record"

    @pytest.mark.asyncio
    async def test_get_memory(self, manager: PersistentMemoryManager) -> None:
        record = _make_record(title="Get me")
        await manager.create(record)
        result = await manager.get(record.id, record.scope)
        assert result is not None
        assert result.id == record.id
        assert result.title == "Get me"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, manager: PersistentMemoryManager) -> None:
        result = await manager.get("nonexistent-id", MemoryScope.USER)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_memory(self, manager: PersistentMemoryManager) -> None:
        record = _make_record(title="Original", body="Original body")
        await manager.create(record)
        original_updated_at = record.updated_at

        updated = await manager.update(record.id, record.scope, {"body": "New body"})
        assert updated is not None
        assert updated.body == "New body"
        assert updated.title == "Original"  # unchanged field preserved
        assert updated.updated_at > original_updated_at

    @pytest.mark.asyncio
    async def test_delete_memory(self, manager: PersistentMemoryManager) -> None:
        record = _make_record()
        await manager.create(record)
        deleted = await manager.delete(record.id, record.scope)
        assert deleted is True
        # Verify it's gone
        result = await manager.get(record.id, record.scope)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(
        self, manager: PersistentMemoryManager
    ) -> None:
        result = await manager.delete("nonexistent-id", MemoryScope.USER)
        assert result is False

    @pytest.mark.asyncio
    async def test_list_by_scope(self, manager: PersistentMemoryManager) -> None:
        r1 = _make_record(title="One", type=MemoryType.USER, scope=MemoryScope.USER)
        r2 = _make_record(title="Two", type=MemoryType.FEEDBACK, scope=MemoryScope.USER)
        r3 = _make_record(
            title="Three", type=MemoryType.PROJECT, scope=MemoryScope.PROJECT
        )
        await manager.create(r1)
        await manager.create(r2)
        await manager.create(r3)

        user_records = await manager.list_by_scope(MemoryScope.USER)
        titles = {r.title for r in user_records}
        assert titles == {"One", "Two"}

        project_records = await manager.list_by_scope(MemoryScope.PROJECT)
        assert len(project_records) == 1
        assert project_records[0].title == "Three"

    @pytest.mark.asyncio
    async def test_list_by_scope_and_type(
        self, manager: PersistentMemoryManager
    ) -> None:
        r1 = _make_record(
            title="User rec", type=MemoryType.USER, scope=MemoryScope.USER
        )
        r2 = _make_record(
            title="Feedback rec", type=MemoryType.FEEDBACK, scope=MemoryScope.USER
        )
        await manager.create(r1)
        await manager.create(r2)

        results = await manager.list_by_scope(
            MemoryScope.USER, memory_type=MemoryType.FEEDBACK
        )
        assert len(results) == 1
        assert results[0].title == "Feedback rec"

    @pytest.mark.asyncio
    async def test_search(self, manager: PersistentMemoryManager) -> None:
        r1 = _make_record(title="Python tips", type=MemoryType.REFERENCE)
        r2 = _make_record(title="Docker notes", type=MemoryType.REFERENCE)
        await manager.create(r1)
        await manager.create(r2)

        # MockStore doesn't do real semantic search, but the API should work
        results = await manager.search("python", MemoryScope.USER)
        # With our mock, asearch returns all items regardless of query
        assert isinstance(results, list)
        assert all(isinstance(r, MemoryRecord) for r in results)

    @pytest.mark.asyncio
    async def test_update_unknown_id_returns_none(
        self, manager: PersistentMemoryManager
    ) -> None:
        """``update`` must not invent records when the id is absent — it returns None."""
        result = await manager.update("does-not-exist", MemoryScope.USER, {"body": "x"})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_relocates_record_when_type_changes(
        self, manager: PersistentMemoryManager, mock_store: Any
    ) -> None:
        """Changing a record's ``type`` moves it to the new type-scoped namespace.

        Without the old-namespace delete the record would exist in two
        places, and reads that probe by type would return duplicates.
        """
        record = _make_record(type=MemoryType.USER, scope=MemoryScope.USER)
        await manager.create(record)
        old_ns = ("memory", MemoryScope.USER.value, MemoryType.USER.value)
        assert record.id in mock_store._data.get(old_ns, {}), (
            "precondition: record should live in the user/user namespace"
        )

        updated = await manager.update(
            record.id,
            record.scope,
            {"type": MemoryType.FEEDBACK.value},
        )
        assert updated is not None
        assert updated.type == MemoryType.FEEDBACK

        new_ns = ("memory", MemoryScope.USER.value, MemoryType.FEEDBACK.value)
        assert record.id in mock_store._data.get(new_ns, {}), (
            "record should now live under the new type's namespace"
        )
        assert record.id not in mock_store._data.get(old_ns, {}), (
            "old namespace should no longer hold the relocated record"
        )

    @pytest.mark.asyncio
    async def test_get_without_type_probes_every_type_namespace(
        self, manager: PersistentMemoryManager
    ) -> None:
        """``get(record_id, scope, memory_type=None)`` searches across all types."""
        record = _make_record(
            title="hide-and-seek",
            type=MemoryType.PROJECT,
            scope=MemoryScope.USER,
        )
        await manager.create(record)

        # Caller doesn't remember which type this record has — pass None
        # and the manager should find it anyway.
        fetched = await manager.get(record.id, record.scope, memory_type=None)
        assert fetched is not None
        assert fetched.type == MemoryType.PROJECT

    @pytest.mark.asyncio
    async def test_list_all_scopes_returns_only_populated_scopes(
        self, manager: PersistentMemoryManager
    ) -> None:
        """Scopes with no records should not appear in the survey."""
        await manager.create(_make_record(scope=MemoryScope.USER, type=MemoryType.USER))
        scopes = await manager.list_all_scopes()
        assert MemoryScope.USER in scopes
        # Scopes we never wrote to should stay out of the result.
        assert MemoryScope.TEAM not in scopes
        assert MemoryScope.ASSISTANT not in scopes

    @pytest.mark.asyncio
    async def test_search_filter_by_type_restricts_namespace_scan(
        self, manager: PersistentMemoryManager
    ) -> None:
        """``search(memory_type=...)`` only hits that type's namespace.

        The search API takes a ``memory_type`` filter that narrows the
        scan to one namespace. This test confirms that records of other
        types aren't returned even though MockStore returns whatever is
        in the namespace it's given.
        """
        await manager.create(_make_record(title="user-typed", type=MemoryType.USER))
        await manager.create(
            _make_record(title="feedback-typed", type=MemoryType.FEEDBACK)
        )
        results = await manager.search(
            "anything", MemoryScope.USER, memory_type=MemoryType.FEEDBACK
        )
        titles = {r.title for r in results}
        assert titles == {"feedback-typed"}, (
            f"search with memory_type=FEEDBACK should only return FEEDBACK"
            f" records; got {titles}"
        )


# ===========================================================================
# SessionNotebook tests (async)
# ===========================================================================


class TestSessionNotebook:
    @pytest.mark.asyncio
    async def test_initialize_creates_notebook(
        self, notebook: SessionNotebook, mock_store: Any
    ) -> None:
        await notebook.initialize()
        item = await mock_store.aget(("session", "test-thread-1"), "notebook")
        assert item is not None
        assert item.value["content"] == NOTEBOOK_TEMPLATE

    @pytest.mark.asyncio
    async def test_load_returns_content(self, notebook: SessionNotebook) -> None:
        await notebook.initialize()
        content = await notebook.load()
        assert "# Session Notebook" in content
        for section in NOTEBOOK_SECTIONS:
            assert f"## {section}" in content

    @pytest.mark.asyncio
    async def test_load_initializes_if_missing(self, notebook: SessionNotebook) -> None:
        # Don't call initialize first — load should handle missing notebook
        content = await notebook.load()
        assert content == NOTEBOOK_TEMPLATE

    @pytest.mark.asyncio
    async def test_update_section(self, notebook: SessionNotebook) -> None:
        await notebook.initialize()
        await notebook.update_section("Current State", "Working on feature X")
        content = await notebook.load()
        assert "Working on feature X" in content
        # Other sections should still be present
        assert "## Task Specification" in content
        assert "## Worklog" in content

    @pytest.mark.asyncio
    async def test_get_section(self, notebook: SessionNotebook) -> None:
        await notebook.initialize()
        await notebook.update_section("Key Results", "Found the bug in line 42")
        result = await notebook.get_section("Key Results")
        assert result == "Found the bug in line 42"

    @pytest.mark.asyncio
    async def test_get_section_not_found(self, notebook: SessionNotebook) -> None:
        await notebook.initialize()
        result = await notebook.get_section("Nonexistent Section")
        assert result == ""

    @pytest.mark.asyncio
    async def test_should_update_thresholds(self, notebook: SessionNotebook) -> None:
        # Below both thresholds
        assert notebook.should_update(0, 0) is False
        assert (
            notebook.should_update(
                DEFAULT_MESSAGE_THRESHOLD - 1, DEFAULT_TOOL_CALL_THRESHOLD - 1
            )
            is False
        )
        # At message threshold
        assert notebook.should_update(DEFAULT_MESSAGE_THRESHOLD, 0) is True
        # At tool call threshold
        assert notebook.should_update(0, DEFAULT_TOOL_CALL_THRESHOLD) is True
        # Both above threshold
        assert (
            notebook.should_update(
                DEFAULT_MESSAGE_THRESHOLD + 5, DEFAULT_TOOL_CALL_THRESHOLD + 5
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_enforce_budget(self, notebook: SessionNotebook) -> None:
        await notebook.initialize()
        # Fill a section with content that exceeds the total budget
        big_content = "X" * (DEFAULT_MAX_TOTAL_TOKENS * 4 + 1000)
        await notebook.update_section("Worklog", big_content)

        total_before = await notebook.get_token_estimate()
        assert total_before > DEFAULT_MAX_TOTAL_TOKENS

        await notebook.enforce_budget()

        # After enforcement, the worklog section should be condensed
        worklog = await notebook.get_section("Worklog")
        assert worklog.startswith("...")
        assert len(worklog) < len(big_content)
