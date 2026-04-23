# Coding Agent

**Source:** `src/langgraph_kit/graphs/coding_agent.py`
**Agent ID:** `coding-agent`

The most capable built-in agent — the reference deep agent with coding-specific overlays.

## Architecture

Extends the reference deep agent with:
- Git context injection
- Coding-specific prompt sections
- Worktree tools
- Enhanced verification (R2-005)

## Additional Prompt Sections

Beyond the reference agent's core sections:

| Section | Content |
|---------|---------|
| Coding Workflow | Step-by-step coding methodology |
| Coding Search | Code search strategies and patterns |
| Worktree Guidance | Git worktree isolation instructions |

## Additional Context Provider

### GitContextProvider

Injects into every prompt:
- Current branch name
- Recent commit messages
- Working tree status (modified/staged files)
- Repository root path

## Additional Tools

### Worktree Tools (R2-004)

Git-specific tools for safe code modification in isolated worktrees. See [Worktree Tools](../tools/worktree-tools.md).

Sourced from `core/tools/worktree.py`, provides: `create_worktree`, `list_worktrees`, `enter_worktree`, `exit_worktree`.

### Enhanced Verifier (R2-005)

Replaces the standard verifier with a coding-specific verification worker (`CODING_VERIFIER_DEFINITION` from `core/orchestration/workers.py`) that:
- Assumes nothing works until evidence proves it does
- Reads actual changed files (not summaries)
- Runs tests if available and reports results
- Reports structured findings: PASS, WARN, FAIL with file paths and line numbers

## Worker Definitions

Pre-composed as `CODING_WORKERS` from `core/orchestration/workers.py`:

| Worker | Definition | Changes from reference agent |
|--------|-----------|-----------------|
| `researcher` | `RESEARCHER_DEFINITION` | Same as reference |
| `implementer` | `IMPLEMENTER_DEFINITION` | Same as reference |
| `verifier` | `CODING_VERIFIER_DEFINITION` | Enhanced skeptical verifier |

See [Workers](../orchestration/workers.md) for full definitions.

## Build Function

```python
def build_coding_agent(
    checkpointer,
    store,
    *,
    mcp_tools=None,
    recursion_limit=DEFAULT_RECURSION_LIMIT,  # 100
):
    """Build the coding agent with reference-agent infrastructure + coding overlays.

    Returns: (compiled_graph, command_dispatcher)
    """
```

## Recursion Limit

Defaults to `DEFAULT_RECURSION_LIMIT` (100). Coding tasks that span many workers, tool calls, and worktree operations can bump into this quickly — raise it for long autonomous runs via `recursion_limit=<n>` at build time, or `config={"recursion_limit": <n>}` per invocation. See the [agents overview](overview.md#recursion-limit) for details.

## When to Use

Use the coding agent when the primary task involves:
- Writing or modifying code
- Debugging and fixing bugs
- Code review and refactoring
- Working with git repositories
- Running tests and build processes
