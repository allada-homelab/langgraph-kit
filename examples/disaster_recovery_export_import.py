"""Disaster recovery: JSON Lines export + selective import.

What this shows
---------------
- Streaming a JSONL export of selected store namespaces via
  :meth:`DisasterRecoveryManager.export_jsonl` (manifest first, then one
  record per line)
- Round-tripping the export back through
  :meth:`DisasterRecoveryManager.import_jsonl` with ``ImportMode.REPLACE``
- Verifying the imported records survive in the destination store

This is **not** a substitute for full database backups — it's a
complement for selective restore (e.g. moving one team's memories
between environments). The schema is versioned (``EXPORT_SCHEMA_VERSION``)
and forward-migration shims are supported via the ``versioned`` mixin.

How to run
----------
    uv run python -m examples.disaster_recovery_export_import

Expected output
---------------
    Source store seeded with 3 records.
    Exported 4 line(s) (1 manifest + 3 records).
    Import result: written=3 skipped=0 namespaces_seen=1
    Destination store now has 3 records under ('memory', 'user').
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, line, make_in_memory_persistence


async def main() -> None:
    banner("disaster_recovery_export_import")

    from langgraph_kit.core.dr import DisasterRecoveryManager, ImportMode

    # 1. Source store: seed three records.
    _, src_store = make_in_memory_persistence()
    seeds = [
        ("user_pref_terse", {"title": "User prefers terse responses"}),
        ("user_pref_pytest", {"title": "Split assertions onto separate lines"}),
        ("user_workflow", {"title": "Always run just pre-commit before push"}),
    ]
    namespace = ("memory", "user")
    for key, value in seeds:
        await src_store.aput(namespace, key, value)
    line(f"Source store seeded with {len(seeds)} records.")

    src_dr = DisasterRecoveryManager(src_store)

    # 2. Stream the export to a list (in production you'd write to a
    #    file or socket — manifest first, then one JSONL line per record).
    exported: list[str] = []
    async for chunk in src_dr.export_jsonl([namespace]):
        exported.append(chunk)
    line(
        f"Exported {len(exported)} line(s) (1 manifest + {len(exported) - 1} records)."
    )

    # 3. Destination store: empty. Import the JSONL stream verbatim.
    _, dst_store = make_in_memory_persistence()
    dst_dr = DisasterRecoveryManager(dst_store)
    result = await dst_dr.import_jsonl(exported, mode=ImportMode.REPLACE)
    line(
        "Import result: "
        f"written={result.written} skipped={result.skipped} "
        f"namespaces_seen={result.namespaces_seen}"
    )

    # 4. Confirm the destination has the same record count.
    items = await dst_store.asearch(namespace, limit=100)
    line(f"Destination store now has {len(items)} records under {namespace}.")


if __name__ == "__main__":
    asyncio.run(main())
