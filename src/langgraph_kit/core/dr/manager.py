"""Point-in-time export / import over a LangGraph ``BaseStore``.

Issue #35. Today the kit relies on the database's own snapshot story
(Postgres, SQLite). That covers full-database recovery but not
selective restore — "give me everything for tenant X" or "ship a
single thread's history to support". This manager fills that gap.

Design:

- **JSON Lines.** One record per line, streamable, appendable,
  readable by ``jq``. The first line is always a manifest with
  ``schema_version`` so a future format change can be rejected
  cleanly.
- **Namespace selection is explicit.** The manager does not enumerate
  every namespace the Store knows about — that requires
  ``alist_namespaces`` support which not every backend exposes
  cheaply. Callers pass the namespace prefixes they want exported;
  the kit ships a sensible default list of "everything Pydantic-
  shaped that we own".
- **Import has three modes.** ``replace`` overwrites the namespace
  before importing; ``append`` only writes records whose ``key``
  doesn't already exist; ``merge`` writes everything (most-recent
  wins on conflicting keys). ``replace`` is the loud default.

Not yet shipped:

- HTTP admin endpoints — the operator workflow needs auth choices
  that aren't kit-internal.
- CLI integration in ``cli.py``.
- Postgres-bypass fast paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable

logger = logging.getLogger(__name__)


EXPORT_SCHEMA_VERSION: int = 1


class ImportMode(StrEnum):
    REPLACE = "replace"  # clear each namespace before importing
    APPEND = "append"  # only add keys that don't already exist
    MERGE = "merge"  # write everything (last wins)


@dataclass
class ExportManifest:
    """First line of an export. Used by import to reject incompatible files."""

    schema_version: int
    exported_at: str
    namespace_prefixes: list[list[str]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": "manifest",
                "schema_version": self.schema_version,
                "exported_at": self.exported_at,
                "namespace_prefixes": self.namespace_prefixes,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExportManifest:
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            exported_at=str(payload.get("exported_at", "")),
            namespace_prefixes=[list(p) for p in payload.get("namespace_prefixes", [])],
        )


@dataclass
class ImportResult:
    """Counts returned by :meth:`DisasterRecoveryManager.import_jsonl`."""

    written: int
    skipped: int
    namespaces_seen: int


class DisasterRecoveryManager:
    """Export / import Store contents as JSON Lines."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self._store = store

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_jsonl(
        self, namespaces: Iterable[tuple[str, ...]]
    ) -> AsyncIterator[str]:
        """Yield manifest + one JSONL line per record across *namespaces*.

        Each yielded string already includes the trailing ``"\\n"``
        — callers can stream directly to a file or socket without
        post-processing.
        """
        prefixes = [list(ns) for ns in namespaces]
        manifest = ExportManifest(
            schema_version=EXPORT_SCHEMA_VERSION,
            exported_at=datetime.now(UTC).isoformat(),
            namespace_prefixes=prefixes,
        )
        yield manifest.to_json() + "\n"

        for ns_prefix in namespaces:
            try:
                items: list[Any] = await self._store.asearch(
                    tuple(ns_prefix), limit=10_000_000
                )
            except Exception:
                logger.exception("DR.export: failed namespace %s", ns_prefix)
                continue
            for item in items:
                payload = {
                    "kind": "record",
                    "namespace": list(ns_prefix),
                    "key": item.key,
                    "value": item.value,
                }
                yield json.dumps(payload, default=str) + "\n"

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    async def import_jsonl(
        self,
        source: AsyncIterable[str] | Iterable[str],
        *,
        mode: ImportMode = ImportMode.REPLACE,
    ) -> ImportResult:
        """Read a JSONL stream and write its records to the Store.

        Iterates manifest first, validates the schema version, then
        processes records line by line. Lines that aren't valid JSON
        or aren't a record-kind payload are logged and skipped.
        """
        manifest_seen = False
        manifest: ExportManifest | None = None
        cleared_namespaces: set[tuple[str, ...]] = set()
        written = 0
        skipped = 0
        namespaces_seen: set[tuple[str, ...]] = set()

        async for line in _to_async_iter(source):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("DR.import: skipping non-JSON line")
                skipped += 1
                continue
            if not isinstance(payload, dict):
                skipped += 1
                continue

            kind = payload.get("kind")

            if kind == "manifest":
                manifest = ExportManifest.from_dict(payload)
                if manifest.schema_version != EXPORT_SCHEMA_VERSION:
                    msg = (
                        f"DR.import: incompatible schema_version "
                        f"{manifest.schema_version} (expected "
                        f"{EXPORT_SCHEMA_VERSION})"
                    )
                    raise ValueError(msg)
                manifest_seen = True
                continue

            if kind != "record":
                skipped += 1
                continue

            if not manifest_seen:
                msg = "DR.import: missing manifest header (first JSONL line)"
                raise ValueError(msg)

            ns = tuple(payload.get("namespace") or ())
            key = payload.get("key")
            value = payload.get("value")
            if not ns or not isinstance(key, str):
                skipped += 1
                continue

            namespaces_seen.add(ns)

            # In replace mode, clear each namespace once on first record.
            if mode == ImportMode.REPLACE and ns not in cleared_namespaces:
                cleared_namespaces.add(ns)
                try:
                    existing: list[Any] = await self._store.asearch(
                        ns, limit=10_000_000
                    )
                    for it in existing:
                        await self._store.adelete(ns, it.key)
                except Exception:
                    logger.exception("DR.import: failed to clear namespace %s", ns)

            # In append mode, skip any key that already exists.
            if mode == ImportMode.APPEND:
                try:
                    existing_item = await self._store.aget(ns, key)
                except Exception:
                    existing_item = None
                if existing_item is not None:
                    skipped += 1
                    continue

            try:
                await self._store.aput(ns, key, value)
                written += 1
            except Exception:
                logger.exception("DR.import: failed to write %s/%s", ns, key)
                skipped += 1

        return ImportResult(
            written=written,
            skipped=skipped,
            namespaces_seen=len(namespaces_seen),
        )


async def _to_async_iter(
    source: AsyncIterable[str] | Iterable[str],
) -> AsyncIterator[str]:
    """Adapt sync iterables to async-iter so import_jsonl can take either."""
    if hasattr(source, "__aiter__"):
        async for line in source:  # type: ignore[union-attr]
            yield line
        return
    for line in source:  # type: ignore[assignment]
        yield line
