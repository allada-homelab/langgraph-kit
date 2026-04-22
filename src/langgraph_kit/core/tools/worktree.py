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

    async def enter_worktree(branch_name: str) -> str:
        """Switch the working directory context to an existing worktree.

        This changes the effective working directory for subsequent git
        operations to the specified worktree.

        Args:
            branch_name: The branch name of the worktree to enter.
        """
        # Verify the worktree exists
        rc, stdout, _ = await _run_git("worktree", "list", "--porcelain", cwd=repo_path)
        if rc != 0:
            return "Error: could not list worktrees."

        target_path: str | None = None
        current_path: str | None = None
        current_branch: str | None = None
        for line in stdout.splitlines():
            if line.startswith("worktree "):
                current_path = line[9:]
            elif line.startswith("branch "):
                branch = line[7:]
                if branch.startswith("refs/heads/"):
                    branch = branch[len("refs/heads/") :]
                current_branch = branch
            elif not line:
                if current_branch == branch_name:
                    target_path = current_path
                current_path = None
                current_branch = None
        # Check last entry
        if current_branch == branch_name:
            target_path = current_path

        if target_path is None:
            return (
                f"Worktree for branch '{branch_name}' not found. "
                f"Create it first with create_worktree."
            )
        return (
            f"Entered worktree at: {target_path} (branch: {branch_name})\n"
            f"Subsequent file operations should target this directory."
        )

    async def exit_worktree(branch_name: str) -> str:
        """Remove a git worktree and clean up its directory.

        This removes the worktree and its associated branch tracking.
        The branch itself is not deleted.

        Args:
            branch_name: The branch name of the worktree to remove.
        """
        rc, _, stderr = await _run_git(
            "worktree",
            "remove",
            f"../{branch_name}",
            cwd=repo_path,
        )
        if rc != 0:
            # Force removal discards uncommitted changes — warn the caller
            logger.warning(
                "Worktree removal failed (%s), retrying with --force. "
                "Uncommitted changes in the worktree will be lost.",
                stderr.strip(),
            )
            dir_name = branch_name.replace("/", "-")
            rc2, _, stderr2 = await _run_git(
                "worktree",
                "remove",
                "--force",
                f"../{dir_name}",
                cwd=repo_path,
            )
            if rc2 != 0:
                return f"Error removing worktree: {stderr2 or stderr}"
            return f"Worktree removed (forced): ../{dir_name}"

        dir_name = branch_name.replace("/", "-")
        return f"Worktree removed: ../{dir_name}"

    return [create_worktree, list_worktrees, enter_worktree, exit_worktree]
