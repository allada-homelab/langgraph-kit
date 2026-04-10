# Context Management Overview

The context management system monitors token consumption, applies mitigation strategies when the context window fills up, and manages continuation decisions for long-running tasks.

## Components

| Module | Purpose |
|--------|---------|
| [Pressure Monitor](pressure.md) | Token estimation and mitigation strategy selection |
| [Compaction](compaction.md) | Conversation summarization (full and partial) |
| [Continuation](continuation.md) | Token-budget continuation with diminishing-returns detection |

## Context Pressure Lifecycle

```
Each agent turn:
    │
    ▼
PressureMonitor.assess(messages)
    │ estimates tokens, counts large outputs
    ▼
PressureMonitor.choose_mitigation(signals)
    │
    ├── < 70% → NONE (no action)
    ├── 70-85% → MICROCOMPACT (truncate old tool outputs)
    ├── 85%+ → FULL_COMPACTION (summarize conversation)
    └── Failed 3x → STOP (circuit breaker)
    │
    ▼
PressureMiddleware applies selected strategy
```

## Mitigation Strategies

| Strategy | Trigger | Action |
|----------|---------|--------|
| `NONE` | Below 70% | No action needed |
| `MICROCOMPACT` | 70-85% | Truncate large tool outputs in older messages |
| `SESSION_ASSISTED` | 70-85% (with notebook) | Use session notebook to guide compaction |
| `FULL_COMPACTION` | Above 85% | LLM-powered conversation summarization |
| `STOP` | 3+ compaction failures | Stop the agent to prevent infinite loops |

## Token Estimation

The pressure monitor uses a simple heuristic: **4 characters ≈ 1 token**. This is fast and good enough for pressure decisions — exact tokenization would be too slow to run every turn.
