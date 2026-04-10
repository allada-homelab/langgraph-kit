# Memory System Overview

The memory system provides persistent, typed, scoped knowledge storage for agents. It enables agents to remember facts across conversations, extract memories automatically, and maintain session-level continuity.

## Components

| Module | Purpose |
|--------|---------|
| [Models](models.md) | `MemoryRecord`, `MemoryType`, `MemoryScope` |
| [Persistent Manager](persistent-manager.md) | CRUD facade over LangGraph Store |
| [Agent Memory](agent-memory.md) | Worker-scoped memory for specialized agents |
| [Auto-Extraction](extraction.md) | LLM-powered post-turn memory extraction |
| [Session Notebook](session-notebook.md) | Thread-local structured notebook |
| [Consolidation](consolidation.md) | Background memory merging, pruning, and normalization |
| [Shared Memory](shared-memory.md) | Team scope publishing with secret detection |

## Memory Taxonomy

Memories are classified by **type** (what kind of information) and **scope** (who it belongs to):

### Types

| Type | Description | Example |
|------|-------------|---------|
| `USER` | Facts about the user | "Senior engineer, prefers Go" |
| `FEEDBACK` | Behavioral guidance | "Don't mock the database in tests" |
| `PROJECT` | Ongoing work context | "Merge freeze starts March 5" |
| `REFERENCE` | Pointers to external resources | "Bugs tracked in Linear project INGEST" |

### Scopes

| Scope | Description |
|-------|-------------|
| `USER` | Belongs to a specific user |
| `ASSISTANT` | Knowledge the assistant has learned |
| `PROJECT` | Shared across the project |
| `TEAM` | Shared across the team |

## Storage Model

All memories are stored in LangGraph's `BaseStore` under hierarchical namespaces:

```
("memory", scope, type)  → individual MemoryRecord items
("memory", "agent", agent_name, type)  → worker-scoped memories
("session", thread_id)  → session notebook
```

Each record is a JSON document with metadata (title, type, scope, timestamps, source) and content (summary, body).

## How Memory Flows

```
User Message
    │
    ▼
Agent processes request
    │
    ▼
Agent may call memory tools ───► PersistentMemoryManager
    (save, list, search,            │
     update, delete)                ▼
    │                          LangGraph Store
    ▼                          (namespace-based)
Agent responds
    │
    ▼
ExtractionMiddleware runs ──► AutoMemoryExtractor
    (post-turn)                    │
    │                              ▼
    │                     LLM identifies durable facts
    │                              │
    │                              ▼
    │                     PersistentMemoryManager
    │                     (create/update/delete)
    ▼
Next turn (memories available via search)
```

## Agent-Callable Memory Tools

When registered, agents get five tools for explicit memory management:

1. **save_memory** — Create a new record with title, type, scope, summary, body
2. **list_memories** — List records by scope, optionally filtered by type
3. **search_memories** — Semantic search across records
4. **update_memory** — Modify an existing record's body or summary
5. **delete_memory** — Remove a record by ID

## Auto-Extraction

The `ExtractionMiddleware` runs after each agent turn and uses an LLM to identify durable facts from the conversation that should be persisted. It avoids duplicating information already stored and respects a taxonomy of what should and should not be saved (e.g., code patterns are derived from the repo, not memorized).

## Consolidation

The `MemoryConsolidator` runs as a background maintenance task, using an LLM to merge near-duplicate records, prune stale entries, and normalize content. See [Consolidation](consolidation.md).

## Shared Memory

The `SharedMemoryManager` enables publishing memories to a team scope for cross-user collaboration, with automatic secret detection that blocks records containing API keys, tokens, or passwords. See [Shared Memory](shared-memory.md).

## Session Notebook

For within-conversation continuity, the `SessionNotebook` maintains structured sections:

- Current State
- Task Specification
- Files and Functions
- Workflow
- Errors and Corrections
- Key Results
- Worklog

The notebook is stored per-thread and helps agents maintain coherent context across long conversations, especially after compaction events.
