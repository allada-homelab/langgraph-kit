# Defining Skills

Skills are defined as `SKILL.md` files with YAML frontmatter and markdown body content.

## File Format

```markdown
---
name: code-review
description: Structured code review with security and quality checklist
tags:
  - quality
  - review
  - security
allowed_tools:
  - search_memories
  - list_memories
---

# Code Review Process

## Step 1: Understand the Change
Read the diff carefully. Identify what changed and why.

## Step 2: Check for Issues
- [ ] Security vulnerabilities (injection, XSS, CSRF)
- [ ] Error handling (edge cases, null checks)
- [ ] Performance (N+1 queries, unnecessary allocations)
- [ ] Code style (naming, structure, readability)

## Step 3: Provide Feedback
...
```

## Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier, 1-64 characters |
| `description` | Yes | What this skill does, max 1024 characters |
| `tags` | No | Searchable labels |
| `allowed_tools` | No | Tools this skill expects to have available |

## Body Content

The body below the frontmatter contains the actual instructions the agent follows. Write it as you would write instructions for a capable colleague:

- Use clear headings for distinct phases
- Include checklists for systematic processes
- Provide examples where helpful
- Reference specific tools the agent should use

## Directory Structure

Place each skill in its own subdirectory:

```
skills/
├── code-review/
│   └── SKILL.md
├── research/
│   └── SKILL.md
└── incident-response/
    └── SKILL.md
```

The `SkillRegistry.load_from_directory()` method recursively scans for all `SKILL.md` files.
