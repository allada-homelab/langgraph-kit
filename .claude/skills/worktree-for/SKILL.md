---
name: worktree-for
description: Create an isolated git worktree + venv for a GitHub issue so the user can run /new-task for it in a parallel Claude session. Setup only, no implementation.
---

# worktree-for

Pure setup. Creates one isolated `git worktree` + venv for an issue, so the user can open a new Claude Code session pointing at it and run [`new-task`](../new-task/SKILL.md) there.

**Invocation:** `/worktree-for <issue-number>`.

Only needed when parallelizing. If only one task is in flight, skip this and run `/new-task` in the primary checkout.

## Setup

```bash
N="$1"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
```

## Pipeline

### 1. Validate the issue

```bash
gh api "repos/$REPO/issues/$N" --jq '{title, state}'
# abort on 404 or if .state != "open"
```

### 2. Derive names

Same slug rule as `new-task`: title lowercased, `[N.M] ` prefix stripped, non-ASCII removed, spaces→dashes, ≤50 chars. Example: issue #17 "[2.2] Structured outputs" → slug `structured-outputs` → branch `feature/17-structured-outputs` → worktree `../lgk-parallel/wt-17`.

```bash
BRANCH="feature/${N}-${SLUG}"
WT_PATH="../lgk-parallel/wt-${N}"
```

### 3. Refuse on pre-existing state

Check three things; abort on any hit with a precise message so the user can act:

```bash
# (a) Worktree for this issue already registered with git.
git worktree list --porcelain | grep -qE "^worktree .*/wt-${N}$" && {
  echo "worktree already exists: $(git worktree list | grep wt-${N})"; exit 1; }

# (b) Target directory exists as non-worktree junk.
test -e "$WT_PATH" && {
  echo "$WT_PATH exists but is not a registered worktree — remove it manually"; exit 1; }

# (c) Branch already exists (leftover from a prior `git worktree remove`).
git show-ref --verify --quiet "refs/heads/$BRANCH" && {
  echo "branch $BRANCH already exists — run \`git branch -D $BRANCH\` if safe to delete"; exit 1; }
```

Explicit abort beats silent force-overwrite. The user sees exactly what to clean.

### 4. Create the worktree off `origin/main`

```bash
git fetch origin
mkdir -p ../lgk-parallel
git worktree add -b "$BRANCH" "$WT_PATH" origin/main
```

### 5. Install dependencies in the worktree

```bash
(cd "$WT_PATH" && just install)
```

Uses the repo's `install` recipe (`uv sync --all-extras --all-groups`). Creates `.venv/` inside the worktree — full isolation from the primary checkout.

### 6. Print the hand-off

Copy-paste block:

```
Worktree ready: ../lgk-parallel/wt-<N>   (branch feature/<N>-<slug>)

To start the task:
  cd ../lgk-parallel/wt-<N>
  claude
Inside the new session:
  /new-task <N>
```

Do **not** `cd` from the current session — the whole point of this skill is isolation.

## Cleanup

Not part of this skill — `new-task` prints the cleanup commands in its success banner once CI is green and the PR is ready to merge:

```bash
git worktree remove ../lgk-parallel/wt-<N>
git branch -d feature/<N>-<slug>   # only after confirming merge
```

## When not to use this skill

- **Single task in flight** → skip straight to `/new-task`.
- **Worktree for this issue already exists** → skill aborts with the existing path.
- **More than 3 worktrees already open** → finish one first. Beyond 3, `detect-conflicts.yml` churns and `pyproject.toml` collisions compound.

## Reporting

Terse. On success, print the hand-off block from step 6. On failure, report which of the three pre-existing-state checks tripped (or which shell step failed) and the exact command to fix it.
