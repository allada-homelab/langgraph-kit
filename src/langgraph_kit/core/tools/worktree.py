"""Worktree isolation tools for coding profiles.

Provides tool functions and a prompt section for managing git worktrees.
Worktrees create isolated copies of the repository for risky or parallel work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Prompt guidance section
# ---------------------------------------------------------------------------

WORKTREE_GUIDANCE_SECTION = PromptSection(
    id="worktree_guidance",
    content=(
        "# Worktree Isolation\n"
        "Use git worktrees when:\n"
        "- The change is experimental or risky and you want to preserve the "
        "current branch state\n"
        "- Multiple independent changes need parallel development\n"
        "- Verification work should run against a separate copy\n\n"
        "Work in-place when:\n"
        "- The change is well-understood and low-risk\n"
        "- You are already on the correct branch\n"
        "- The task is a single logical commit\n\n"
        "Always clean up worktrees when done."
    ),
    stability=SectionStability.STABLE,
    priority=40,
)


# ---------------------------------------------------------------------------
# Git subprocess helper
# ---------------------------------------------------------------------------


async def _run_git(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT_SECONDS
        )
        return (
            proc.returncode or 0,
            stdout.decode().strip(),
            stderr.decode().strip(),
        )
    except TimeoutError:
        return 1, "", "Command timed out"
    except (FileNotFoundError, OSError) as exc:
        return 1, "", str(exc)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def build_worktree_tools(repo_path: str | None = None) -> list[Any]:
    """Create tool functions for managing git worktrees.

    Returns a list of async callables suitable for registration as ToolCapability.
    """

    async def create_worktree(branch_name: str, base_ref: str = "HEAD") -> str:
        """Create an isolated git worktree for a new branch.

        Creates a new worktree directory alongside the main repo, checked out
        to a new branch based on the given ref.

        Args:
            branch_name: Name for the new branch in the worktree.
            base_ref: Git ref to base the new branch on (default: HEAD).
        """
        # Sanitize branch name for directory path (branches like "feature/foo"
        # would create nested dirs)
        dir_name = branch_name.replace("/", "-")
        rc, stdout, stderr = await _run_git(
            "worktree",
            "add",
            "-b",
            branch_name,
            f"../{dir_name}",
            base_ref,
            cwd=repo_path,
        )
        if rc != 0:
            return f"Error creating worktree: {stderr}"
        return f"Worktree created: ../{dir_name} (branch: {branch_name})\n{stdout}"

    async def list_worktrees() -> str:
        """List all active git worktrees.

        Returns the list of worktrees with their branches and HEAD commits.
        """
        rc, stdout, stderr = await _run_git(
            "worktree", "list", "--porcelain", cwd=repo_path
        )
        if rc != 0:
            return f"Error listing worktrees: {stderr}"
        if not stdout:
            return "No worktrees found."

        # Parse porcelain output into human-readable format
        entries: list[str] = []
        current: dict[str, str] = {}
        for line in stdout.splitlines():
            if not line:
                if current:
                    path = current.get("worktree", "?")
                    branch = current.get("branch", "detached")
                    # Shorten refs/heads/ prefix
                    if branch.startswith("refs/heads/"):
                        branch = branch[len("refs/heads/") :]
                    entries.append(f"- {path} (branch: {branch})")
                    current = {}
            elif line.startswith("worktree "):
                current["worktree"] = line[9:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]
        # Handle last entry without trailing newline
        if current:
            path = current.get("worktree", "?")
            branch = current.get("branch", "detached")
            if branch.startswith("refs/heads/"):
                branch = branch[len("refs/heads/") :]
            entries.append(f"- {path} (branch: {branch})")

        return f"Active worktrees ({len(entries)}):\n" + "\n".join(entries)

    async def exit_worktree(branch_name: str, force: bool = False) -> str:
        """Remove a git worktree and clean up its directory.

        This removes the worktree and its associated branch tracking.
        The branch itself is not deleted.

        Args:
            branch_name: The branch name of the worktree to remove.
            force: When True, pass ``--force`` to git worktree remove.
                Destructive: uncommitted changes in the worktree are lost.
                Off by default so the tool refuses to delete dirty trees
                silently.
        """
        git_args: list[str] = ["worktree", "remove"]
        if force:
            git_args.append("--force")
        dir_name = branch_name.replace("/", "-")
        git_args.append(f"../{dir_name}")

        rc, _, stderr = await _run_git(*git_args, cwd=repo_path)
        if rc != 0:
            hint = ""
            if not force and "contains modified" in stderr.lower():
                hint = (
                    " (uncommitted changes present — re-invoke with force=True "
                    "to discard them)"
                )
            return f"Error removing worktree: {stderr}{hint}"
        prefix = "Worktree removed (forced)" if force else "Worktree removed"
        return f"{prefix}: ../{dir_name}"

    # ``enter_worktree`` was a read-only stub: it returned a path string
    # with no way for downstream tools to honour the "new cwd". Rather
    # than keep a misleading tool, rely on ``list_worktrees`` +
    # ``create_worktree`` for the discovery/creation story and delete the
    # stub. If a future cwd-plumbing story lands, this tool can come back.

    return [create_worktree, list_worktrees, exit_worktree]
