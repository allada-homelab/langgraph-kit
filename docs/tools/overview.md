# Tools & Capabilities Overview

The tool system provides a rich capability model that goes beyond simple function registration. Each tool carries metadata about its risk level, applicable profiles, prompt guidance, and more — enabling intelligent filtering and compilation.

## Components

| Module | Purpose |
|--------|---------|
| [Tool Capability](capability.md) | `ToolCapability` dataclass and `ToolRisk` enum |
| [Tool Registry](registry.md) | Registration, filtering, compilation |
| [Memory Tools](memory-tools.md) | Agent-callable CRUD tools for memory |
| [Deferred Tools](deferred-tools.md) | Lazy tool discovery and registration |

## Risk Taxonomy

Every tool has a risk level that controls filtering and HITL behavior:

| Risk Level | Description | Example |
|------------|-------------|---------|
| `READ_ONLY` | No side effects | `search_memories`, `list_tools` |
| `MUTATING` | Modifies state | `save_memory`, `update_memory` |
| `DESTRUCTIVE` | Hard to reverse | `delete_memory`, file deletion |

## Tool Lifecycle

```
1. Define ToolCapability
       │
       ▼
2. Register in ToolRegistry
       │
       ▼
3. Filter by profile/worker/tags/risk
       │
       ▼
4. Compile to callable functions
       │
       ▼
5. Bind to LLM as tools
```

## Filtering Dimensions

Tools can be filtered by:

- **Profile** — e.g., `"coding"`, `"research"`, `"general"`
- **Worker type** — e.g., `"researcher"`, `"implementer"`, `"verifier"`
- **Tags** — arbitrary string labels
- **Max risk** — e.g., only `READ_ONLY` for coordinator mode

## Prompt Fragments

Tools can carry `prompt_guidance` — a text fragment that gets injected into the system prompt to help the agent use the tool effectively. The registry's `collect_prompt_fragments()` method gathers these for prompt assembly.
