"""Git-aware context provider for coding profiles (R2-002).

Injects current branch, changed-file summary, and repository cleanliness
into the prompt at composition time. Only added to coding-profile agents.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.0


async def _run_git(*args: str, cwd: str | None = None) -> str:
    """Run a git command and return stripped stdout, or empty string on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            return ""
        return stdout.decode().strip()
    except (TimeoutError, FileNotFoundError, OSError):
        return ""


class GitContextProvider:
    """Injects git repository state into the prompt.

    Provides: current branch, changed-file summary, clean/dirty status.
    Returns empty string when git is unavailable or the directory is not a repo.
    """

    def __init__(self, repo_path: str | None = None) -> None:
        super().__init__()
        self._repo_path = repo_path

    async def provide(self, context: dict[str, Any]) -> str:
        cwd = self._repo_path or context.get("repo_path")

        # Verify we're in a git repo
        toplevel = await _run_git("rev-parse", "--show-toplevel", cwd=cwd)
        if not toplevel:
            return ""

        branch, status, diff_stat = await asyncio.gather(
            _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
            _run_git("status", "--porcelain", cwd=cwd),
            _run_git("diff", "--stat", "HEAD", cwd=cwd),
        )

        if not branch:
            return ""

        lines = ["# Repository State"]
        lines.append(f"Branch: `{branch}`")

        if status:
            changed_files = [line[3:] for line in status.splitlines() if len(line) > 3]
            lines.append(f"Status: **{len(changed_files)} changed file(s)**")
            # Show up to 15 files
            for f in changed_files[:15]:
                lines.append(f"  - {f}")
            if len(changed_files) > 15:
                lines.append(f"  - ... and {len(changed_files) - 15} more")
        else:
            lines.append("Status: **clean**")

        if diff_stat:
            # Last line of diff --stat is the summary (e.g. "3 files changed, ...")
            summary_line = diff_stat.splitlines()[-1].strip()
            if summary_line:
                lines.append(f"Diff: {summary_line}")

        return "\n".join(lines)
