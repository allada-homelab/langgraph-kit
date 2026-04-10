# Session Notebook

**Source:** `src/langgraph_kit/core/memory/session.py`

A thread-local structured notebook that maintains continuity within a conversation, especially useful across compaction events.

## Class: SessionNotebook

### Constructor

```python
SessionNotebook(store: BaseStore, thread_id: str)
```

Stored at namespace `("session", thread_id)`, key `"notebook"`.

### Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Create notebook from template if it doesn't exist |
| `load()` | Load current notebook content |
| `save(content)` | Overwrite entire notebook |
| `update_section(name, content)` | Replace content of a specific section |
| `get_section(name)` | Extract content of a specific section |
| `should_update(messages)` | Decide if notebook should update based on activity |
| `estimate_tokens(content)` | Rough token estimate (4 chars per token) |
| `condense_section(name, max_tokens)` | Truncate section if it exceeds budget |
| `enforce_budget(max_total)` | Condense sections if total exceeds budget |

### Notebook Sections

| Section | Purpose |
|---------|---------|
| Current State | Where the conversation stands right now |
| Task Specification | What the user asked for |
| Files and Functions | Key files and code locations discussed |
| Workflow | Steps taken or planned |
| Errors and Corrections | Problems encountered and how they were resolved |
| Key Results | Important outputs or findings |
| Worklog | Chronological activity log |

### Update Thresholds

The notebook updates when activity exceeds thresholds:

| Constant | Default | Trigger |
|----------|---------|---------|
| `DEFAULT_MESSAGE_THRESHOLD` | 6 | Messages since last update |
| `DEFAULT_TOOL_CALL_THRESHOLD` | 4 | Tool calls since last update |

### Token Budgets

| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_MAX_SECTION_TOKENS` | 500 | Max tokens per section |
| `DEFAULT_MAX_TOTAL_TOKENS` | 3,000 | Max tokens for entire notebook |

When budgets are exceeded, `enforce_budget()` condenses sections by truncation.

## Usage

The session notebook is primarily used by deep agents for maintaining context across long conversations:

```python
notebook = SessionNotebook(store, thread_id)
await notebook.initialize()

# After significant work
await notebook.update_section("Current State", "Implementing auth middleware")
await notebook.update_section("Files and Functions", "- auth/middleware.py:42\n- tests/test_auth.py")

# Before compaction
content = await notebook.load()
# Content is included in compaction prompt for better summaries
```
