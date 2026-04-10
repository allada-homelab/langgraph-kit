# R0 Agent

**Source:** `src/langgraph_kit/graphs/r0_agent.py`
**Agent ID:** `r0-agent`

The full-featured general-purpose agent demonstrating all R0-level capabilities.

## Architecture

Built on the deepagents framework with the complete langgraph-kit feature stack:

```
User Message
    │
    ▼
Middleware Stack (11 middleware)
    │
    ▼
deepagents create_deep_agent()
    ├── Prompt Assembly (PromptComposer)
    ├── Tool Registry (standard + memory + UI + HITL)
    ├── Worker Definitions (researcher, implementer, verifier)
    ├── Backend Factory (memories, notes, state)
    └── Command Dispatcher (7 built-in commands)
```

## Prompt Sections

The R0 agent registers 5 core prompt sections:

| Section ID | Stability | Content |
|-----------|-----------|---------|
| `core_identity` | STABLE | Agent identity and capabilities |
| `memory_instructions` | STABLE | How to use persistent memory |
| `orchestration_instructions` | STABLE | Multi-agent delegation patterns |
| `continuation_guidance` | STABLE | When to continue vs. stop |
| `ui_interaction` | CONDITIONAL | UI artifact and event usage |

## Worker Definitions

Three specialized workers for task delegation, defined in `core/orchestration/workers.py`:

| Worker | Definition | Description |
|--------|-----------|-------------|
| `researcher` | `RESEARCHER_DEFINITION` | Finds information, reads docs, searches code |
| `implementer` | `IMPLEMENTER_DEFINITION` | Writes code, makes changes, builds features |
| `verifier` | `VERIFIER_DEFINITION` | Reviews changes, runs tests, validates output |

Pre-composed as `R0_WORKERS` — see [Workers](../orchestration/workers.md).

## Features Included

- Prompt assembly with stable-first caching
- Persistent memory (CRUD + auto-extraction)
- Tool registry with risk levels
- 7 built-in slash commands
- Context pressure monitoring
- Session notebook continuity
- Async task delegation
- Message queue for busy threads
- HITL approval for destructive operations
- UI artifacts, progress, suggestions, citations
- Skill discovery
- MCP tool integration
- Resilience middleware (completion guard, empty turn, tool error)
- Langfuse observability

## Build Function

```python
def build_r0_agent(checkpointer, store, mcp_tools=None):
    """Build the R0 agent with all features.

    Returns: (compiled_graph, command_dispatcher)
    """
```

Returns a tuple of the compiled graph and command dispatcher, both registered together.

## Middleware Stack

The R0 agent uses the full standard middleware stack (11 middleware), built by `core.graph_builder.build_middleware_stack()`:

1. CommandMiddleware
2. RuntimeStateMiddleware
3. QueuedInputMiddleware
4. ToolErrorMiddleware
5. PressureMiddleware
6. ResultPersistenceMiddleware
7. ExtractionMiddleware
8. EmptyTurnMiddleware
9. CompletionGuardMiddleware
10. StopHooksMiddleware
11. PostRunBackstopMiddleware
