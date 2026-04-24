---
name: changelog-entry
description: Draft an entry for `## [Unreleased]` in CHANGELOG.md, matching the project's Keep-a-Changelog style. Use after implementing a feature or fix, before opening the PR.
---

# changelog-entry

Append a properly-categorized entry to the `## [Unreleased]` section of [CHANGELOG.md](../../../CHANGELOG.md). The repo follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## Workflow

1. **Survey the change.** Prefer `git diff main...HEAD` (full branch) over staged-only — the entry should cover what will be in the PR, not just the working tree.
2. **Pick a bucket** under `## [Unreleased]`:
   - **Added** — new features, modules, public API.
   - **Changed** — behavior changes to existing features.
   - **Fixed** — bug fixes.
   - **Removed** — deletions of public API.
   - **Deprecated** — features marked for removal.
   - **Security** — vulnerability fixes.
3. **Match the existing voice.** Read the most recent few entries first. The convention is:
   - Lead with a **bold one-liner** stating what changed.
   - Follow with one paragraph explaining the *why* and any user-visible impact (migration notes, new options, breaking changes).
   - Cross-link docs (`[docs/path.md](docs/path.md)`) and source (`[src/.../file.py](src/.../file.py)`) when relevant.
4. **Insert under the right subheading.** Create the subheading if it doesn't exist yet in `## [Unreleased]`.
5. **Show the diff** before applying — let the user confirm tone and scope.

## Don't

- Don't move items out of `[Unreleased]` into a versioned `## [X.Y.Z] — date` section. That happens at release time per [CONTRIBUTING.md](../../../CONTRIBUTING.md#releasing).
- Don't add internal-only refactors with no user-visible impact.
- Don't duplicate an existing `[Unreleased]` line — extend or amend it instead.
- Don't include emojis (the repo enforces no-emojis in user-facing output and the changelog follows the same convention).
