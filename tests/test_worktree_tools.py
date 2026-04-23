"""Coverage fill — worktree tool bodies with mocked ``_run_git``.

``build_worktree_tools`` returns four async tools that shell out to
``git worktree …`` via the module-level ``_run_git`` helper. Existing
tests only check the tools exist and are callable; these unit tests
drive each tool body with a mocked ``_run_git`` so the subprocess-free
branches (porcelain parsing, error paths, force-remove fallback) are
exercised.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from langgraph_kit.core.tools import worktree as worktree_mod
from langgraph_kit.core.tools.worktree import build_worktree_tools


@pytest.mark.asyncio
async def test_create_worktree_success() -> None:
    tools: list[Any] = build_worktree_tools()
    create = next(t for t in tools if t.__name__ == "create_worktree")

    with patch.object(
        worktree_mod,
        "_run_git",
        new=AsyncMock(return_value=(0, "Preparing worktree", "")),
    ):
        result = await create("feature/x", "main")

    assert "Worktree created" in result
    # Slash in branch name should be sanitized in the dir path.
    assert "feature-x" in result


@pytest.mark.asyncio
async def test_create_worktree_error_path_surfaces_stderr() -> None:
    tools: list[Any] = build_worktree_tools()
    create = next(t for t in tools if t.__name__ == "create_worktree")

    with patch.object(
        worktree_mod,
        "_run_git",
        new=AsyncMock(return_value=(1, "", "fatal: branch 'x' already exists")),
    ):
        result = await create("x")

    assert "Error creating worktree" in result
    assert "already exists" in result


@pytest.mark.asyncio
async def test_list_worktrees_parses_porcelain_output() -> None:
    tools: list[Any] = build_worktree_tools()
    listw = next(t for t in tools if t.__name__ == "list_worktrees")

    porcelain = (
        "worktree /repo\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /repo-alt\n"
        "branch refs/heads/feature/x\n"
        "\n"
    )
    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(0, porcelain, ""))
    ):
        result = await listw()

    # Both worktrees surface with branch names stripped of refs/heads/.
    assert "Active worktrees (2)" in result
    assert "/repo (branch: main)" in result
    assert "/repo-alt (branch: feature/x)" in result


@pytest.mark.asyncio
async def test_list_worktrees_reports_empty() -> None:
    tools: list[Any] = build_worktree_tools()
    listw = next(t for t in tools if t.__name__ == "list_worktrees")

    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(0, "", ""))
    ):
        assert await listw() == "No worktrees found."


@pytest.mark.asyncio
async def test_list_worktrees_error_path() -> None:
    tools: list[Any] = build_worktree_tools()
    listw = next(t for t in tools if t.__name__ == "list_worktrees")

    with patch.object(
        worktree_mod,
        "_run_git",
        new=AsyncMock(return_value=(1, "", "not a git repo")),
    ):
        result = await listw()

    assert "Error listing worktrees" in result
    assert "not a git repo" in result


@pytest.mark.asyncio
async def test_enter_worktree_locates_branch_in_porcelain_output() -> None:
    tools: list[Any] = build_worktree_tools()
    enter = next(t for t in tools if t.__name__ == "enter_worktree")

    porcelain = (
        "worktree /repo\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /other\n"
        "branch refs/heads/target\n"
    )
    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(0, porcelain, ""))
    ):
        result = await enter("target")

    assert "/other" in result
    assert "branch: target" in result


@pytest.mark.asyncio
async def test_enter_worktree_branch_not_found() -> None:
    tools: list[Any] = build_worktree_tools()
    enter = next(t for t in tools if t.__name__ == "enter_worktree")

    porcelain = "worktree /repo\nbranch refs/heads/main\n"
    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(0, porcelain, ""))
    ):
        result = await enter("nowhere")

    assert "not found" in result
    assert "create_worktree" in result


@pytest.mark.asyncio
async def test_enter_worktree_git_error_surfaces() -> None:
    tools: list[Any] = build_worktree_tools()
    enter = next(t for t in tools if t.__name__ == "enter_worktree")

    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(1, "", "bad"))
    ):
        assert (
            "could not list worktrees" in (await enter("any")).lower()
        )


@pytest.mark.asyncio
async def test_exit_worktree_happy_path() -> None:
    tools: list[Any] = build_worktree_tools()
    exitw = next(t for t in tools if t.__name__ == "exit_worktree")

    with patch.object(
        worktree_mod, "_run_git", new=AsyncMock(return_value=(0, "", ""))
    ):
        result = await exitw("feature-x")

    assert "Worktree removed" in result
    assert "feature-x" in result


@pytest.mark.asyncio
async def test_exit_worktree_falls_back_to_force_remove() -> None:
    tools: list[Any] = build_worktree_tools()
    exitw = next(t for t in tools if t.__name__ == "exit_worktree")

    # First call fails (uncommitted changes), forced call succeeds.
    call_log: list[tuple[str, ...]] = []

    async def fake_run_git(*args: str, cwd: str | None = None) -> Any:
        _ = cwd
        call_log.append(args)
        if "--force" in args:
            return (0, "", "")
        return (1, "", "contains uncommitted changes")

    with patch.object(worktree_mod, "_run_git", new=fake_run_git):
        result = await exitw("dirty")

    assert "forced" in result.lower()
    assert len(call_log) == 2, (
        f"Expected two git calls (initial + --force retry); got {call_log}"
    )
    assert "--force" in call_log[1]


@pytest.mark.asyncio
async def test_exit_worktree_force_remove_also_fails() -> None:
    tools: list[Any] = build_worktree_tools()
    exitw = next(t for t in tools if t.__name__ == "exit_worktree")

    async def always_fail(*args: str, cwd: str | None = None) -> Any:
        _ = args
        _ = cwd
        return (1, "", "permanent error")

    with patch.object(worktree_mod, "_run_git", new=always_fail):
        result = await exitw("dead")

    assert "Error removing worktree" in result
    assert "permanent error" in result


@pytest.mark.asyncio
async def test_run_git_handles_missing_binary_gracefully() -> None:
    """``_run_git`` returns ``(1, "", exc_message)`` when git isn't on PATH."""

    async def raise_file_not_found(
        *args: str, **kwargs: Any
    ) -> Any:
        _ = args
        _ = kwargs
        msg = "no git"
        raise FileNotFoundError(msg)

    with patch.object(
        worktree_mod.asyncio, "create_subprocess_exec", new=raise_file_not_found
    ):
        rc, stdout, stderr = await worktree_mod._run_git("status")

    assert rc == 1
    assert stdout == ""
    assert "no git" in stderr
