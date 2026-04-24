# Resilience Overview

The resilience system provides middleware that catches failure modes and prevents the agent from getting stuck, producing empty output, or stopping prematurely.

## Components

| Module | Purpose |
|--------|---------|
| [Completion Guard](completion-guard.md) | Heuristic premature-completion detection |
| [Empty Turn](empty-turn.md) | Nudging the model when no output is produced |
| [Tool Error](tool-error.md) | Structured error handling with transient retry |
| [Post-Run Backstop](post-run.md) | Final checks after graph execution |
| [Graceful Shutdown](graceful-shutdown.md) | Drain in-flight async sub-agent tasks on SIGTERM |

## Middleware Stack Position

Resilience middleware is positioned in the standard stack to catch different failure modes at different stages:

```
1. CommandMiddleware         ← command interception
2. QueuedInputMiddleware     ← inject queued messages
3. ToolErrorMiddleware       ← wrap tool calls (catch errors)
4. PressureMiddleware        ← context management
5. ResultPersistenceMiddleware
6. ExtractionMiddleware
7. EmptyTurnMiddleware       ← detect empty output
8. CompletionGuardMiddleware ← detect premature completion
9. StopHooksMiddleware
10. PostRunBackstopMiddleware ← final safety check
```

## Design Philosophy

Rather than letting failures propagate as unhandled exceptions or silent empty responses, resilience middleware converts failures into structured feedback that the agent can act on:

- Tool errors become structured error messages the agent can reason about
- Empty turns become nudges asking the agent to take concrete action
- Premature completions become challenges asking the agent to justify stopping
- All guards have maximum attempt limits to prevent infinite loops
