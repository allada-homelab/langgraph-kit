# Worktree Tools

**Source:** `src/langgraph_kit/core/tools/worktree.py`

Git worktree isolation tools for coding profiles (R2-004). Provides tool functions and a prompt section for managing git worktrees — isolated copies of the repository for risky or parallel work.

## build_worktree_tools(repo_path=None)

Returns a list of 4 async tool functions:

### create_worktree(branch_name, base_ref="HEAD")

Create an isolated git worktree for a new branch.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branch_name` | `str` | _(required)_ | Name for the new branch |
| `base_ref` | `str` | `"HEAD"` | Git ref to base the branch on |

Creates a worktree directory alongside the main repo at `../{branch_name}`.

### list_worktrees()

List all active git worktrees with their branches and paths. Parses `git worktree list --porcelain` into a human-readable format.

### enter_worktree(branch_name)

Switch the working directory context to an existing worktree. Verifies the worktree exists before reporting the path.

### exit_worktree(branch_name)

Remove a git worktree and clean up its directory. Attempts force removal if there are uncommitted changes. The branch itself is not deleted.

## WORKTREE_GUIDANCE_SECTION

A `PromptSection` (stability: STABLE, priority: 40) that guides the agent on when to use worktrees:

**Use worktrees when:**
- The change is experimental or risky
- Multiple independent changes need parallel development
- Verification work should run against a separate copy

**Work in-place when:**
- The change is well-understood and low-risk
- Already on the correct branch
- The task is a single logical commit

## Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `_TIMEOUT_SECONDS` | 10.0 | Timeout for git subprocess calls |

## Usage

```python
from langgraph_kit.core.tools.worktree import build_worktree_tools, WORKTREE_GUIDANCE_SECTION

# Register tools
tools = build_worktree_tools(repo_path="/path/to/repo")
for tool_fn in tools:
    register_tool(registry, tool_fn, id_prefix="worktree", tags=["git", "coding"])

# Add prompt section
section_registry.register(WORKTREE_GUIDANCE_SECTION)
```
