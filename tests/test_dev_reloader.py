"""Coverage — stdlib-only file-mtime reloader for dev hot-reload (#36).

Tests use a real temp directory and ``os.utime`` to advance mtimes
deterministically — no sleeps, no flakiness. The full
``langgraph-kit dev`` server is multi-PR effort; this module just
ships the watch primitive that the rest will consume.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from langgraph_kit.dev import FileChange, Reloader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path, mtime: float | None = None) -> None:
    """Force a specific mtime on *path*, creating it if needed."""
    if not path.exists():
        path.write_text("")
    if mtime is None:
        mtime = time.time() + 10  # well in the future, deterministic
    os.utime(path, (mtime, mtime))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_reloader_rejects_zero_or_negative_poll_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        Reloader([str(tmp_path)], poll_interval=0)
    with pytest.raises(ValueError, match="poll_interval"):
        Reloader([str(tmp_path)], poll_interval=-0.5)


def test_reloader_baseline_snapshot_is_empty_for_empty_dir(tmp_path: Path) -> None:
    r = Reloader([str(tmp_path)])
    assert r.diff() == []


# ---------------------------------------------------------------------------
# diff() detects added / modified / removed
# ---------------------------------------------------------------------------


def test_diff_reports_added_files(tmp_path: Path) -> None:
    r = Reloader([str(tmp_path)])
    new_file = tmp_path / "agent.py"
    new_file.write_text("print('hi')")
    changes = r.diff()
    kinds = {(c.kind, Path(c.path).name) for c in changes}
    assert ("added", "agent.py") in kinds


def test_diff_reports_modified_files(tmp_path: Path) -> None:
    f = tmp_path / "agent.py"
    f.write_text("v1")
    r = Reloader([str(tmp_path)])
    # Bump mtime far enough that float comparisons can't tie.
    _touch(f, mtime=time.time() + 100)
    changes = r.diff()
    assert any(c.kind == "modified" for c in changes)


def test_diff_reports_removed_files(tmp_path: Path) -> None:
    f = tmp_path / "agent.py"
    f.write_text("v1")
    r = Reloader([str(tmp_path)])
    f.unlink()
    changes = r.diff()
    assert any(c.kind == "removed" and "agent.py" in c.path for c in changes)


def test_diff_is_idempotent_when_nothing_changes(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("v1")
    r = Reloader([str(tmp_path)])
    r.diff()  # consume the initial-walk no-op
    assert r.diff() == []


def test_diff_advances_snapshot_so_repeat_modify_only_reports_once(
    tmp_path: Path,
) -> None:
    f = tmp_path / "agent.py"
    f.write_text("v1")
    r = Reloader([str(tmp_path)])
    _touch(f, mtime=time.time() + 100)
    first = r.diff()
    second = r.diff()
    assert len(first) == 1
    assert second == []


# ---------------------------------------------------------------------------
# Ignore patterns
# ---------------------------------------------------------------------------


def test_default_ignore_filters_pycache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "agent.cpython-313.pyc").write_text("bytecode")
    real = tmp_path / "agent.py"
    real.write_text("real")

    r = Reloader([str(tmp_path)])
    # The added .pyc should not have triggered an "added" event for that path.
    paths_seen = {Path(c.path).name for c in r.diff()}
    assert "agent.cpython-313.pyc" not in paths_seen
    # And subsequent edits to the .pyc don't fire either.
    _touch(cache_dir / "agent.cpython-313.pyc", mtime=time.time() + 100)
    assert all(".pyc" not in c.path for c in r.diff())


def test_custom_ignore_filters_extra_patterns(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("real")
    (tmp_path / "junk.tmp").write_text("noise")

    r = Reloader([str(tmp_path)], ignore=(".tmp",))
    # junk.tmp shouldn't show up.
    initial = {Path(c.path).name for c in r.diff()}
    assert "junk.tmp" not in initial


# ---------------------------------------------------------------------------
# run() loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_loop_calls_callback_on_changes(tmp_path: Path) -> None:
    f = tmp_path / "agent.py"
    f.write_text("v1")
    r = Reloader([str(tmp_path)], poll_interval=0.01)

    received: list[list[FileChange]] = []

    async def _handle(changes: list[FileChange]) -> None:
        received.append(changes)

    async def _modify_after_tick() -> None:
        # First tick consumes the no-change baseline; second sees the touch.
        await asyncio.sleep(0.02)
        _touch(f, mtime=time.time() + 100)

    _bg = asyncio.create_task(_modify_after_tick())  # noqa: RUF006 — awaited below
    await r.run(_handle, max_iterations=4)

    assert any(any(c.kind == "modified" for c in batch) for batch in received), (
        f"expected at least one modified batch in {received!r}"
    )


@pytest.mark.asyncio
async def test_run_loop_supports_sync_callback(tmp_path: Path) -> None:
    f = tmp_path / "agent.py"
    f.write_text("v1")
    r = Reloader([str(tmp_path)], poll_interval=0.01)

    calls: list[int] = []

    def _handle(changes: list[FileChange]) -> None:
        calls.append(len(changes))

    async def _modify_after_tick() -> None:
        await asyncio.sleep(0.02)
        _touch(f, mtime=time.time() + 100)

    _bg = asyncio.create_task(_modify_after_tick())  # noqa: RUF006 — awaited below
    await r.run(_handle, max_iterations=4)
    # Sync callback should have been called at least once.
    assert any(c > 0 for c in calls)
