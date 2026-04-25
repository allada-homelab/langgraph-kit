"""Coverage — DisasterRecoveryManager export / import (#35).

Stream contract: first JSONL line is a manifest with
``schema_version``, subsequent lines are records of the form
``{"kind": "record", "namespace": [...], "key": "...", "value": ...}``.

Import modes verified independently:

- ``replace`` clears each visited namespace before importing.
- ``append`` keeps existing keys, only writes new ones.
- ``merge`` writes everything, last value wins on conflicts.
"""

from __future__ import annotations

import json

import pytest

from langgraph_kit.core.dr import (
    EXPORT_SCHEMA_VERSION,
    DisasterRecoveryManager,
    ImportMode,
)
from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed(store: MockStore) -> None:
    await store.aput(("memory", "user", "feedback"), "m1", {"id": "m1", "title": "ok"})
    await store.aput(
        ("memory", "user", "feedback"), "m2", {"id": "m2", "title": "neat"}
    )
    await store.aput(("threads", "t-1"), "metadata", {"thread_id": "t-1"})


async def _collect_jsonl(mgr: DisasterRecoveryManager, namespaces) -> list[str]:
    lines: list[str] = []
    async for line in mgr.export_jsonl(namespaces):
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_starts_with_manifest_then_records() -> None:
    store = MockStore()
    await _seed(store)
    mgr = DisasterRecoveryManager(store)

    lines = await _collect_jsonl(
        mgr, [("memory", "user", "feedback"), ("threads", "t-1")]
    )
    assert len(lines) == 4  # 1 manifest + 2 memory + 1 thread
    manifest = json.loads(lines[0])
    assert manifest["kind"] == "manifest"
    assert manifest["schema_version"] == EXPORT_SCHEMA_VERSION
    # Record lines have shape {kind:record, namespace, key, value}.
    for raw in lines[1:]:
        record = json.loads(raw)
        assert record["kind"] == "record"
        assert isinstance(record["namespace"], list)
        assert isinstance(record["key"], str)
        assert "value" in record


@pytest.mark.asyncio
async def test_export_handles_empty_namespace_gracefully() -> None:
    store = MockStore()
    mgr = DisasterRecoveryManager(store)
    lines = await _collect_jsonl(mgr, [("does", "not", "exist")])
    # Just the manifest — no records.
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "manifest"


@pytest.mark.asyncio
async def test_export_lines_are_newline_terminated() -> None:
    """Streamable: callers can pipe directly to a file without joining."""
    store = MockStore()
    await _seed(store)
    mgr = DisasterRecoveryManager(store)
    lines = await _collect_jsonl(mgr, [("memory", "user", "feedback")])
    assert all(line.endswith("\n") for line in lines)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_trip_export_then_import_into_fresh_store() -> None:
    src = MockStore()
    await _seed(src)
    mgr_src = DisasterRecoveryManager(src)
    lines = await _collect_jsonl(
        mgr_src, [("memory", "user", "feedback"), ("threads", "t-1")]
    )

    dst = MockStore()
    mgr_dst = DisasterRecoveryManager(dst)
    result = await mgr_dst.import_jsonl(lines, mode=ImportMode.REPLACE)

    # Same record count; namespaces equivalent.
    assert result.written == 3
    assert result.skipped == 0
    item = await dst.aget(("memory", "user", "feedback"), "m1")
    assert item is not None
    assert item.value["title"] == "ok"


@pytest.mark.asyncio
async def test_import_replace_clears_namespace_first() -> None:
    src = MockStore()
    await _seed(src)
    mgr_src = DisasterRecoveryManager(src)
    lines = await _collect_jsonl(mgr_src, [("memory", "user", "feedback")])

    dst = MockStore()
    # Pre-populate with stale records — replace must clear them.
    await dst.aput(("memory", "user", "feedback"), "stale", {"junk": True})
    mgr_dst = DisasterRecoveryManager(dst)
    await mgr_dst.import_jsonl(lines, mode=ImportMode.REPLACE)

    assert await dst.aget(("memory", "user", "feedback"), "stale") is None
    assert await dst.aget(("memory", "user", "feedback"), "m1") is not None


@pytest.mark.asyncio
async def test_import_append_skips_existing_keys() -> None:
    src = MockStore()
    await _seed(src)
    mgr_src = DisasterRecoveryManager(src)
    lines = await _collect_jsonl(mgr_src, [("memory", "user", "feedback")])

    dst = MockStore()
    await dst.aput(("memory", "user", "feedback"), "m1", {"title": "original"})
    mgr_dst = DisasterRecoveryManager(dst)
    result = await mgr_dst.import_jsonl(lines, mode=ImportMode.APPEND)

    assert result.skipped >= 1
    # Original record untouched.
    item = await dst.aget(("memory", "user", "feedback"), "m1")
    assert item is not None
    assert item.value["title"] == "original"
    # New record was added.
    assert await dst.aget(("memory", "user", "feedback"), "m2") is not None


@pytest.mark.asyncio
async def test_import_merge_overwrites_existing_keys() -> None:
    src = MockStore()
    await _seed(src)
    mgr_src = DisasterRecoveryManager(src)
    lines = await _collect_jsonl(mgr_src, [("memory", "user", "feedback")])

    dst = MockStore()
    await dst.aput(("memory", "user", "feedback"), "m1", {"title": "stale"})
    mgr_dst = DisasterRecoveryManager(dst)
    await mgr_dst.import_jsonl(lines, mode=ImportMode.MERGE)

    item = await dst.aget(("memory", "user", "feedback"), "m1")
    assert item is not None
    assert item.value["title"] == "ok"  # merge wrote the new value


@pytest.mark.asyncio
async def test_import_rejects_missing_manifest() -> None:
    """An import without the manifest header must fail loudly — no
    silent recovery on an unknown format."""
    dst = MockStore()
    mgr = DisasterRecoveryManager(dst)
    lines_no_manifest = [
        json.dumps({"kind": "record", "namespace": ["a"], "key": "k", "value": 1})
        + "\n"
    ]
    with pytest.raises(ValueError, match="manifest"):
        await mgr.import_jsonl(lines_no_manifest)


@pytest.mark.asyncio
async def test_import_rejects_incompatible_schema_version() -> None:
    dst = MockStore()
    mgr = DisasterRecoveryManager(dst)
    bad_manifest = (
        json.dumps(
            {
                "kind": "manifest",
                "schema_version": 999,
                "exported_at": "2026-04-25",
                "namespace_prefixes": [],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="schema_version"):
        await mgr.import_jsonl([bad_manifest])


@pytest.mark.asyncio
async def test_import_skips_non_json_and_non_record_lines() -> None:
    dst = MockStore()
    mgr = DisasterRecoveryManager(dst)
    lines = [
        json.dumps(
            {
                "kind": "manifest",
                "schema_version": EXPORT_SCHEMA_VERSION,
                "exported_at": "now",
                "namespace_prefixes": [],
            }
        ),
        "not-json-at-all",
        json.dumps({"kind": "comment", "text": "ignore me"}),
        json.dumps({"kind": "record", "namespace": ["a"], "key": "k", "value": 1}),
    ]
    result = await mgr.import_jsonl(lines, mode=ImportMode.MERGE)
    assert result.written == 1
    assert result.skipped == 2


@pytest.mark.asyncio
async def test_import_accepts_async_iterable_source() -> None:
    """The source param accepts async iterables (e.g. file streaming)."""
    dst = MockStore()
    mgr = DisasterRecoveryManager(dst)

    async def _stream():
        yield (
            json.dumps(
                {
                    "kind": "manifest",
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "exported_at": "x",
                    "namespace_prefixes": [],
                }
            )
            + "\n"
        )
        yield (
            json.dumps({"kind": "record", "namespace": ["x"], "key": "k", "value": "v"})
            + "\n"
        )

    result = await mgr.import_jsonl(_stream(), mode=ImportMode.MERGE)
    assert result.written == 1
