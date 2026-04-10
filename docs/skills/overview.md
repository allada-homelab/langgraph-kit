# Skills Overview

The skills system provides progressive disclosure of specialized capabilities through SKILL.md files. Rather than loading all instructions into the system prompt upfront, skills are discovered on demand when the agent needs them.

## Components

| Module | Purpose |
|--------|---------|
| [Skill Registry](registry.md) | Discovery, indexing, and search |
| [Defining Skills](defining-skills.md) | Writing SKILL.md files |

## How Skills Work

```
Agent receives request about code review
    │
    ▼
Agent calls: discover_skills("code review")
    │
    ▼
SkillRegistry.search("code review")
    │ matches: "code-review" skill
    ▼
Agent calls: get_skill_guidance("code-review")
    │
    ▼
SkillRegistry.get_full_content("code-review")
    │ returns SKILL.md body content
    ▼
Agent follows the detailed instructions
```

## Benefits

- **Reduced prompt size** — only load instructions when needed
- **Modular** — each skill is a self-contained document
- **Discoverable** — agents can search for relevant skills
- **Extensible** — add new skills by dropping SKILL.md files

## Agent-Callable Tools

When registered, agents get two tools:

| Tool | Description |
|------|-------------|
| `discover_skills` | Search for relevant skills by keyword |
| `get_skill_guidance` | Load full instructions for a specific skill |
