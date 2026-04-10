# Coding Agent

**Source:** `src/langgraph_kit/graphs/coding_agent.py`
**Agent ID:** `coding-agent`

The most capable built-in agent — R0 with coding-specific overlays.

## Architecture

Extends the R0 agent with:
- Git context injection
- Coding-specific prompt sections
- Worktree tools
- Enhanced verification (R2-005)

## Additional Prompt Sections

Beyond R0's core sections:

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

Replaces the standard R0 verifier with a coding-specific verification worker (`CODING_VERIFIER_DEFINITION` from `core/orchestration/workers.py`) that:
- Assumes nothing works until evidence proves it does
- Reads actual changed files (not summaries)
- Runs tests if available and reports results
- Reports structured findings: PASS, WARN, FAIL with file paths and line numbers

## Worker Definitions

Pre-composed as `CODING_WORKERS` from `core/orchestration/workers.py`:

| Worker | Definition | Changes from R0 |
|--------|-----------|-----------------|
| `researcher` | `RESEARCHER_DEFINITION` | Same as R0 |
| `implementer` | `IMPLEMENTER_DEFINITION` | Same as R0 |
| `verifier` | `CODING_VERIFIER_DEFINITION` | Enhanced skeptical verifier |

See [Workers](../orchestration/workers.md) for full definitions.

## Build Function

```python
def build_coding_agent(checkpointer, store, mcp_tools=None):
    """Build the coding agent with R0 + coding overlays.

    Returns: (compiled_graph, command_dispatcher)
    """
```

## When to Use

Use the coding agent when the primary task involves:
- Writing or modifying code
- Debugging and fixing bugs
- Code review and refactoring
- Working with git repositories
- Running tests and build processes
