"""Dev: stdlib-only file-mtime watcher used by ``langgraph-kit dev``.

What this shows
---------------
- :class:`Reloader` polling a temp directory at a fast cadence
- :class:`FileChange` records ("added" / "modified" / "removed")
  produced by :meth:`Reloader.diff` between polls
- The default ignore patterns filtering out ``__pycache__``, ``.pyc``,
  build caches, and VCS dirs

The reloader is the foundation for the longer hot-reload story (graph
rebuild + checkpoint preservation + inspector UI) tracked in #36. This
demo exercises just the polling primitive.

How to run
----------
    uv run python -m examples.dev_hot_reload

Expected output
---------------
    Watching /tmp/lgk-example-XXXX
    --- after writing agent.py ---
    1 change(s):
      added    agent.py
    --- after touching agent.py ---
    1 change(s):
      modified agent.py
    --- after deleting agent.py ---
    1 change(s):
      removed  agent.py
"""

from __future__ import annotations

import asyncio
import os
import time

from examples._lib import banner, line, tmp_workspace


async def main() -> None:
    banner("dev_hot_reload")

    from langgraph_kit.dev import Reloader

    with tmp_workspace() as workspace:
        line(f"Watching {workspace}")

        reloader = Reloader([str(workspace)], poll_interval=0.05)

        # 1. Add a file.
        target = workspace / "agent.py"
        target.write_text("print('hi')\n")
        # Push the mtime to a deterministic future point so the diff
        # comparison is stable across filesystems with low-res timestamps.
        future = time.time() + 10
        os.utime(target, (future, future))
        changes = reloader.diff()
        line("--- after writing agent.py ---")
        line(f"{len(changes)} change(s):")
        for c in changes:
            line(f"  {c.kind:<8} {target.relative_to(workspace)}")

        # 2. Touch (modify mtime).
        future += 5
        os.utime(target, (future, future))
        changes = reloader.diff()
        line("--- after touching agent.py ---")
        line(f"{len(changes)} change(s):")
        for c in changes:
            line(f"  {c.kind:<8} {target.relative_to(workspace)}")

        # 3. Remove.
        target.unlink()
        changes = reloader.diff()
        line("--- after deleting agent.py ---")
        line(f"{len(changes)} change(s):")
        for c in changes:
            line(f"  {c.kind:<8} {target.name}")


if __name__ == "__main__":
    asyncio.run(main())
