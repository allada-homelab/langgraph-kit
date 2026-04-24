---
name: code-review
description: |
  Use when reviewing code changes, PRs, or diffs. Provides a structured
  review workflow with severity levels. Do NOT use for writing new code.
tags: [review, quality, pr]
---

# Code Review Skill

## Workflow

1. **Understand scope**: Read the diff or list of changed files to understand what changed and why.
2. **Check each file**: For every changed file, read the relevant sections and evaluate against the criteria below.
3. **Report findings**: Produce a structured report with severity levels.

## Review Criteria

- **Correctness**: Does the code do what it claims? Are edge cases handled?
- **Security**: Any injection risks, hardcoded secrets, or OWASP top-10 issues?
- **Performance**: Unnecessary allocations, N+1 queries, missing indexes?
- **Style**: Does it follow existing project conventions?
- **Tests**: Are changes covered by tests? Are test assertions meaningful?

## Severity Levels

- **CRITICAL**: Must fix before merge — security vulnerabilities, data loss risk, broken functionality
- **WARNING**: Should fix — performance issues, missing edge cases, style violations that hurt readability
- **INFO**: Optional — suggestions for improvement, alternative approaches, minor nits

## Output Format

```
## Code Review: [file or PR title]

### CRITICAL
- [file:line] Description of issue

### WARNING
- [file:line] Description of issue

### INFO
- [file:line] Suggestion

### Summary
[1-2 sentence overall assessment]
```
