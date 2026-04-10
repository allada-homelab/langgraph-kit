# Orchestration Overview

The orchestration system provides multi-agent coordination patterns: async sub-agents for background work, message queues for busy threads, and a coordinator profile for read-only delegation.

## Components

| Module | Purpose |
|--------|---------|
| [Workers](workers.md) | Declarative worker (sub-agent) definitions |
| [Async Tasks](async-tasks.md) | Fire-and-forget background sub-agents |
| [Message Queue](queue.md) | Store-backed per-thread message queue |
| [Coordinator](coordinator.md) | Supervisor profile with read-only delegation |

## Patterns

### Delegation

The primary agent can delegate work to specialized [workers](workers.md) (researcher, implementer, verifier) via async tasks. Worker definitions are declared in `core/orchestration/workers.py` as dicts compatible with deepagents' `subagents` parameter. Each worker runs in its own thread with its own checkpointer state.

```
Primary Agent (user-facing)
    │
    ├── start_async_task("researcher", "Find API documentation")
    │       → runs in background thread
    │
    ├── start_async_task("implementer", "Build auth middleware")
    │       → runs in background thread
    │
    └── check_async_task(task_id)
            → polls for completion
```

### Queue Buffering

When a thread is actively processing, new user messages are buffered in the queue. The `QueuedInputMiddleware` drains the queue at the start of each turn, injecting queued messages as additional context.

```
User sends message while agent is busy
    │
    ▼
POST /queue → ThreadQueue.enqueue()
    │ (message buffered)
    ▼
Agent finishes current turn
    │
    ▼
QueuedInputMiddleware.abefore_model()
    │ drains queue, injects messages
    ▼
Agent processes both original + queued messages
```

### Coordinator Mode

A read-only supervisor that can observe and delegate but not directly modify state. Used for oversight patterns where one agent coordinates others.
