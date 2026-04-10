# Skill Registry

**Source:** `src/langgraph_kit/core/skills/registry.py`

Discovers, indexes, and serves SKILL.md files.

## Class: SkillRegistry

### Methods

| Method | Description |
|--------|-------------|
| `load_from_directory(path)` | Scan directory tree for SKILL.md files |
| `get(name)` | Retrieve skill metadata by name |
| `list_all()` | Return all loaded skills |
| `get_full_content(name)` | Return the body content of a SKILL.md |
| `search(query)` | Keyword search across names, descriptions, tags |
| `build_catalog_prompt()` | Build a prompt section listing available skills |

### load_from_directory(path)

Recursively scans the directory for `SKILL.md` files. Each file is parsed for YAML frontmatter to extract `SkillMetadata`.

Expected directory structure:
```
skills/
├── code-review/
│   └── SKILL.md
├── research/
│   └── SKILL.md
└── deployment/
    └── SKILL.md
```

### search(query)

Performs keyword matching against:
- Skill name
- Skill description
- Skill tags

Returns matching `SkillMetadata` objects sorted by relevance.

### build_catalog_prompt()

Generates a formatted list of available skills for inclusion in the system prompt:

```
Available skills:
- code-review: Structured code review process with checklist [tags: quality, review]
- research: Deep research methodology with source verification [tags: research, analysis]
```

## SkillMetadata

**Source:** `src/langgraph_kit/core/skills/models.py`

```python
class SkillMetadata(BaseModel):
    name: str                      # 1-64 characters
    description: str               # Max 1024 characters
    path: str                      # File path to SKILL.md
    tags: list[str] = []           # Searchable tags
    allowed_tools: list[str] = []  # Tools this skill needs
```
