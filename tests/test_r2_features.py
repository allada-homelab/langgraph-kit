"""Tests for R2 coding-profile overlay features."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_kit.core.commands.builtins import (
    build_context_command,
    build_help_command,
    build_memory_command,
)
from langgraph_kit.core.commands.dispatch import (
    CommandDispatcher,
    CommandResult,
)
from langgraph_kit.core.orchestration.verification import (
    CODING_VERIFIER_DEFINITION,
)
from langgraph_kit.core.prompt_assembly.coding_sections import (
    CODING_SEARCH_SECTIONS,
    CODING_WORKFLOW_SECTIONS,
)
from langgraph_kit.core.prompt_assembly.git_context import (
    GitContextProvider,
)
from langgraph_kit.core.prompt_assembly.sections import (
    SectionRegistry,
    SectionStability,
)
from langgraph_kit.core.tools.worktree import (
    WORKTREE_GUIDANCE_SECTION,
    build_worktree_tools,
)


def test_coding_workflow_sections_are_stable() -> None:
    for section in CODING_WORKFLOW_SECTIONS:
        assert section.stability == SectionStability.STABLE


def test_coding_search_sections_are_stable() -> None:
    for section in CODING_SEARCH_SECTIONS:
        assert section.stability == SectionStability.STABLE


def test_coding_sections_have_unique_ids() -> None:
    all_sections = CODING_WORKFLOW_SECTIONS + CODING_SEARCH_SECTIONS
    ids = [s.id for s in all_sections]
    assert len(ids) == len(set(ids)), f"Duplicate section IDs: {ids}"


def test_coding_sections_included_in_registry() -> None:
    registry = SectionRegistry()
    registry.register_many(CODING_WORKFLOW_SECTIONS)
    registry.register_many(CODING_SEARCH_SECTIONS)

    # All coding sections are STABLE, so they should be active with no conditions
    active = registry.get_active(conditions=set())
    active_ids = {s.id for s in active}
    assert "coding_workflow_rules" in active_ids
    assert "coding_search_first" in active_ids


def test_coding_sections_do_not_conflict_with_core() -> None:
    """Coding sections should not overwrite core sections."""
    from langgraph_kit.graphs.r0_agent import _CORE_SECTIONS

    registry = SectionRegistry()
    registry.register_many(_CORE_SECTIONS)
    registry.register_many(CODING_WORKFLOW_SECTIONS)
    registry.register_many(CODING_SEARCH_SECTIONS)

    # All sections should coexist
    core_ids = {s.id for s in _CORE_SECTIONS}
    coding_ids = {s.id for s in CODING_WORKFLOW_SECTIONS + CODING_SEARCH_SECTIONS}
    assert core_ids.isdisjoint(coding_ids), (
        "Coding sections must not shadow core sections"
    )


def test_coding_sections_priority_below_core() -> None:
    """Coding overlay sections should have lower priority than core sections."""
    from langgraph_kit.graphs.r0_agent import _CORE_SECTIONS

    max_coding_priority = max(
        s.priority for s in CODING_WORKFLOW_SECTIONS + CODING_SEARCH_SECTIONS
    )
    min_core_priority = min(s.priority for s in _CORE_SECTIONS)
    assert max_coding_priority < min_core_priority


# ---------------------------------------------------------------------------
# 2. GitContextProvider (R2-002)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_context_provider_returns_branch() -> None:
    provider = GitContextProvider()

    async def mock_run(*args: str, cwd: str | None = None) -> str:
        cmd = " ".join(args)
        if "rev-parse --show-toplevel" in cmd:
            return "/repo"
        if "rev-parse --abbrev-ref HEAD" in cmd:
            return "feature/my-branch"
        if "status --porcelain" in cmd:
            return ""
        if "diff --stat" in cmd:
            return ""
        return ""

    with patch(
        "langgraph_kit.core.prompt_assembly.git_context._run_git",
        side_effect=mock_run,
    ):
        result = await provider.provide({})

    assert "feature/my-branch" in result
    assert "clean" in result.lower()


@pytest.mark.asyncio
async def test_git_context_provider_returns_changed_files() -> None:
    provider = GitContextProvider()

    async def mock_run(*args: str, cwd: str | None = None) -> str:
        cmd = " ".join(args)
        if "rev-parse --show-toplevel" in cmd:
            return "/repo"
        if "rev-parse --abbrev-ref HEAD" in cmd:
            return "main"
        if "status --porcelain" in cmd:
            return " M src/app.py\n?? new_file.txt"
        if "diff --stat" in cmd:
            return " src/app.py | 5 ++---\n 1 file changed, 2 insertions(+), 3 deletions(-)"
        return ""

    with patch(
        "langgraph_kit.core.prompt_assembly.git_context._run_git",
        side_effect=mock_run,
    ):
        result = await provider.provide({})

    assert "2 changed file(s)" in result
    assert "src/app.py" in result
    assert "new_file.txt" in result


@pytest.mark.asyncio
async def test_git_context_provider_handles_no_git() -> None:
    provider = GitContextProvider()

    async def mock_run(*args: str, cwd: str | None = None) -> str:
        return ""  # All git commands fail

    with patch(
        "langgraph_kit.core.prompt_assembly.git_context._run_git",
        side_effect=mock_run,
    ):
        result = await provider.provide({})

    assert result == ""


# ---------------------------------------------------------------------------
# 3. Worktree Tools (R2-004)
# ---------------------------------------------------------------------------


def test_build_worktree_tools_returns_four() -> None:
    tools = build_worktree_tools()
    assert len(tools) == 4


def test_worktree_tools_are_callable() -> None:
    tools = build_worktree_tools()
    for tool in tools:
        assert callable(tool)


def test_worktree_tools_have_names() -> None:
    tools = build_worktree_tools()
    names = [getattr(t, "__name__", None) for t in tools]
    assert "create_worktree" in names
    assert "list_worktrees" in names
    assert "enter_worktree" in names
    assert "exit_worktree" in names


def test_worktree_guidance_section_is_stable() -> None:
    assert WORKTREE_GUIDANCE_SECTION.stability == SectionStability.STABLE
    assert WORKTREE_GUIDANCE_SECTION.id == "worktree_guidance"


@pytest.mark.asyncio
async def test_create_worktree_runs_git() -> None:
    tools = build_worktree_tools()
    create = tools[0]  # create_worktree

    with patch(
        "langgraph_kit.core.tools.worktree._run_git",
        new_callable=AsyncMock,
        return_value=(0, "Preparing worktree", ""),
    ) as mock_git:
        result = await create("test-branch")

    assert "test-branch" in result
    mock_git.assert_called_once()
    call_args = mock_git.call_args[0]
    assert "worktree" in call_args
    assert "add" in call_args


@pytest.mark.asyncio
async def test_list_worktrees_parses_output() -> None:
    tools = build_worktree_tools()
    list_wt = tools[1]  # list_worktrees

    porcelain = (
        "worktree /repo\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /repo/../feature\n"
        "branch refs/heads/feature\n"
        "\n"
    )
    with patch(
        "langgraph_kit.core.tools.worktree._run_git",
        new_callable=AsyncMock,
        return_value=(0, porcelain, ""),
    ):
        result = await list_wt()

    assert "main" in result
    assert "feature" in result
    assert "2" in result


# ---------------------------------------------------------------------------
# 4. Verification Worker (R2-005)
# ---------------------------------------------------------------------------


def test_coding_verifier_has_structured_output_format() -> None:
    prompt = CODING_VERIFIER_DEFINITION["system_prompt"]
    assert "PASS" in prompt
    assert "WARN" in prompt
    assert "FAIL" in prompt


def test_coding_verifier_is_independent() -> None:
    prompt = CODING_VERIFIER_DEFINITION["system_prompt"]
    assert "Do NOT fix" in prompt


def test_coding_verifier_has_required_keys() -> None:
    assert "name" in CODING_VERIFIER_DEFINITION
    assert "description" in CODING_VERIFIER_DEFINITION
    assert "system_prompt" in CODING_VERIFIER_DEFINITION
    assert CODING_VERIFIER_DEFINITION["name"] == "verifier"


# ---------------------------------------------------------------------------
# 5. Slash-Command Dispatch (R2-006)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_registers_command() -> None:
    dispatcher = CommandDispatcher()

    async def handler(args: str, ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(output="ok")

    dispatcher.register("test", handler, description="A test command")
    assert dispatcher.is_command("/test")
    assert not dispatcher.is_command("/unknown")
    assert not dispatcher.is_command("not a command")


@pytest.mark.asyncio
async def test_dispatcher_dispatch_known_command() -> None:
    dispatcher = CommandDispatcher()

    async def handler(args: str, ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(output=f"received: {args}")

    dispatcher.register("echo", handler)
    result = await dispatcher.dispatch("/echo hello world")

    assert result.handled is True
    assert result.output == "received: hello world"


@pytest.mark.asyncio
async def test_dispatcher_unknown_command() -> None:
    dispatcher = CommandDispatcher()
    result = await dispatcher.dispatch("/unknown")

    assert result.handled is False
    assert "Unknown command" in result.output


@pytest.mark.asyncio
async def test_dispatcher_not_a_command() -> None:
    dispatcher = CommandDispatcher()
    result = await dispatcher.dispatch("regular text")

    assert result.handled is False


def test_dispatcher_list_commands() -> None:
    dispatcher = CommandDispatcher()

    async def noop(args: str, ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(output="")

    dispatcher.register("help", noop, description="Show help")
    dispatcher.register("status", noop, description="Show status")

    commands = dispatcher.list_commands()
    assert len(commands) == 2
    names = {c.name for c in commands}
    assert names == {"help", "status"}


@pytest.mark.asyncio
async def test_dispatcher_case_insensitive() -> None:
    dispatcher = CommandDispatcher()

    async def handler(args: str, ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(output="ok")

    dispatcher.register("Test", handler)
    assert dispatcher.is_command("/test")
    assert dispatcher.is_command("/TEST")

    result = await dispatcher.dispatch("/TEST")
    assert result.handled is True


# ---------------------------------------------------------------------------
# 6. Built-in Commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_command_lists_all() -> None:
    dispatcher = CommandDispatcher()

    async def noop(args: str, ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(output="")

    dispatcher.register("memory", noop, description="Inspect memories")
    dispatcher.register("context", noop, description="Context status")

    help_handler = build_help_command(dispatcher)
    # Register help itself
    dispatcher.register("help", help_handler, description="Show help")

    result = await help_handler("", {})
    assert "memory" in result.output
    assert "context" in result.output
    assert "help" in result.output


@pytest.mark.asyncio
async def test_memory_command_shows_records(mock_store: Any) -> None:
    from langgraph_kit.core.memory.models import (
        MemoryRecord,
        MemoryScope,
        MemoryType,
    )
    from langgraph_kit.core.memory.persistent import (
        PersistentMemoryManager,
    )

    mgr = PersistentMemoryManager(mock_store)
    record = MemoryRecord(
        title="Test Memory",
        type=MemoryType.PROJECT,
        scope=MemoryScope.USER,
        summary="A test memory",
        body="details",
    )
    await mgr.create(record)

    handler = build_memory_command(mgr)
    result = await handler("user", {})

    assert "Test Memory" in result.output
    assert result.metadata["count"] == 1


@pytest.mark.asyncio
async def test_memory_command_invalid_scope() -> None:
    mgr = MagicMock()
    handler = build_memory_command(mgr)
    result = await handler("invalid_scope", {})
    assert "Invalid scope" in result.output


@pytest.mark.asyncio
async def test_context_command_shows_status() -> None:
    from langgraph_kit.core.context_management.pressure import (
        PressureMonitor,
    )

    monitor = PressureMonitor(window_limit=100_000)
    handler = build_context_command(monitor)
    result = await handler("", {"messages": []})

    assert "Context Window Status" in result.output
    assert "100,000" in result.output
    assert "estimated_tokens" in result.metadata


# ---------------------------------------------------------------------------
# 7. Coding Worker Definitions
# ---------------------------------------------------------------------------


def test_coding_worker_definitions_has_three() -> None:
    from langgraph_kit.core.orchestration.workers import CODING_WORKERS

    assert len(CODING_WORKERS) == 3
    names = [d["name"] for d in CODING_WORKERS]
    assert "researcher" in names
    assert "implementer" in names
    assert "verifier" in names


def test_coding_verifier_differs_from_r0() -> None:
    from langgraph_kit.core.orchestration.workers import CODING_WORKERS, R0_WORKERS

    r0_verifier = next(d for d in R0_WORKERS if d["name"] == "verifier")
    coding_verifier = next(
        d for d in CODING_WORKERS if d["name"] == "verifier"
    )

    # The coding verifier should have the enhanced system prompt
    assert "PASS" in coding_verifier["system_prompt"]
    assert "PASS" not in r0_verifier["system_prompt"]


# ---------------------------------------------------------------------------
# 8. build_coding_agent smoke test
# ---------------------------------------------------------------------------


def test_build_coding_agent_returns_graph(mock_store: Any) -> None:
    """Call build_coding_agent with mock store + mock checkpointer."""
    from langgraph_kit.graphs.coding_agent import build_coding_agent

    checkpointer = MagicMock()
    fake_graph = MagicMock(name="compiled_graph")
    deepagents_mod = MagicMock()
    deepagents_mod.create_deep_agent.return_value = fake_graph
    fake_llm = MagicMock(name="fake_llm")

    backends_mod = MagicMock()
    module_patches = {
        "deepagents": deepagents_mod,
        "deepagents.backends": backends_mod,
        "deepagents.backends.composite": backends_mod.composite,
        "deepagents.backends.state": backends_mod.state,
        "deepagents.backends.store": backends_mod.store,
    }

    with (
        patch.dict(sys.modules, module_patches),
        patch("langgraph_kit.graphs._builder.build_llm", return_value=fake_llm),
    ):
        graph, _dispatcher = build_coding_agent(checkpointer=checkpointer, store=mock_store)

    assert graph is fake_graph
    deepagents_mod.create_deep_agent.assert_called_once()

    # Verify it was called with coding-specific args
    call_kwargs = deepagents_mod.create_deep_agent.call_args
    assert call_kwargs[1]["name"] == "coding-agent" or (len(call_kwargs[0]) > 0)
