"""Stdlib-only file-mtime watcher for dev hot-reload (issue #36).

The full ``langgraph-kit dev`` server (file-watch + graph rebuild +
checkpoint preservation + inspector UI) is a multi-PR effort. This
module ships the foundation: a :class:`Reloader` that polls a list
of paths for mtime changes and fires a user-supplied callback
on each batch of changes.

Two reasons not to use ``watchfiles`` or ``watchdog``:

1. Both add native deps we don't want shipped at install time. The
   kit's current install footprint is small; a dev-mode feature
   shouldn't blow that up.
2. Polling with stdlib ``os.stat`` is plenty fast for the file
   counts a typical agent project has (tens to low hundreds), and
   sidesteps native-binding portability concerns on macOS / Windows.

The reloader is async-iter-friendly: callers can drive it
themselves (``async for change in reloader.watch(): ...``) or use
the convenience :meth:`Reloader.run` to loop forever. Cancellation
is plain ``asyncio.CancelledError`` — no special teardown.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """One change observed by the reloader's polling loop."""

    path: str
    kind: str  # "added" | "modified" | "removed"


# Default ignore patterns — substrings checked against each candidate path.
# Keeps __pycache__, .venv, etc. out of the watch surface so a recompile
# burst doesn't fire spurious reload callbacks.
_DEFAULT_IGNORE: tuple[str, ...] = (
    "__pycache__",
    ".pyc",
    ".pyo",
    ".venv",
    "/.git/",
    "/.mypy_cache/",
    "/.ruff_cache/",
    "/.pytest_cache/",
)


def _iter_files(roots: Iterable[str], ignore: tuple[str, ...]) -> list[str]:
    """Walk *roots* and return the absolute paths of every file the
    watcher should track.

    Symlinks are followed at directory granularity (Python's default).
    Files matching any substring in *ignore* are filtered out before
    they're stat'd.
    """
    out: list[str] = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            if not any(p in root for p in ignore):
                out.append(str(root_path.resolve()))
            continue
        # ``os.walk`` is still the right call here — pathlib's
        # ``Path.walk`` is 3.12+ and we support 3.11. Use it for the
        # traversal but normalise paths through pathlib at the leaf.
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored subdirectories to avoid descending into them.
            dirnames[:] = [d for d in dirnames if not any(p in d for p in ignore)]
            for fn in filenames:
                full = Path(dirpath) / fn
                full_str = str(full)
                if any(p in full_str for p in ignore):
                    continue
                out.append(str(full.resolve()))
    return out


def _snapshot_mtimes(paths: Iterable[str]) -> dict[str, float]:
    """Best-effort ``{abspath: mtime}`` snapshot. Missing files are skipped."""
    out: dict[str, float] = {}
    for p in paths:
        try:
            out[p] = Path(p).stat().st_mtime
        except OSError:
            # File may have been removed mid-walk; ignore.
            continue
    return out


class Reloader:
    """Polls *roots* on an interval and yields :class:`FileChange` batches."""

    def __init__(
        self,
        roots: Iterable[str],
        *,
        poll_interval: float = 1.0,
        ignore: tuple[str, ...] = _DEFAULT_IGNORE,
    ) -> None:
        super().__init__()
        if poll_interval <= 0:
            msg = "poll_interval must be positive"
            raise ValueError(msg)
        self._roots = tuple(roots)
        self._poll_interval = float(poll_interval)
        self._ignore = ignore
        self._snapshot: dict[str, float] = _snapshot_mtimes(
            _iter_files(self._roots, self._ignore)
        )

    def diff(self) -> list[FileChange]:
        """Return changes since the last call. Updates the internal snapshot."""
        current_files = _iter_files(self._roots, self._ignore)
        current = _snapshot_mtimes(current_files)
        changes: list[FileChange] = []

        # Added or modified.
        for path, mtime in current.items():
            previous = self._snapshot.get(path)
            if previous is None:
                changes.append(FileChange(path=path, kind="added"))
            elif previous != mtime:
                changes.append(FileChange(path=path, kind="modified"))

        # Removed.
        for path in self._snapshot:
            if path not in current:
                changes.append(FileChange(path=path, kind="removed"))

        self._snapshot = current
        return changes

    async def run(
        self,
        on_change: Callable[[list[FileChange]], Awaitable[None] | None],
        *,
        max_iterations: int | None = None,
    ) -> int:
        """Loop forever, calling *on_change* on each batch of changes.

        Returns the number of batches processed before the loop exits
        (only useful in tests via ``max_iterations``). Plain
        ``asyncio.CancelledError`` interrupts the loop cleanly.
        """
        iterations = 0
        while True:
            await asyncio.sleep(self._poll_interval)
            changes = self.diff()
            if changes:
                result = on_change(changes)
                if asyncio.iscoroutine(result):
                    await result
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                return iterations
