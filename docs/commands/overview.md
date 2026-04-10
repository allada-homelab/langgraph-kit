# Command System Overview

The command system provides a transport-independent slash-command dispatch mechanism. Users type `/command` in the chat, and the middleware intercepts it before the LLM sees it, routing to the appropriate handler.

## Components

| Module | Purpose |
|--------|---------|
| [Dispatcher](dispatcher.md) | `CommandDispatcher` routing and execution |
| [Middleware](middleware.md) | `CommandMiddleware` for pre-LLM interception |
| [Built-in Commands](builtins.md) | `/help`, `/memory`, `/context`, `/compact`, `/status`, `/tools`, `/skills` |

## Architecture

```
User types "/compact"
    │
    ▼
CommandMiddleware.abefore_agent()
    │ extracts command from last user message
    ▼
CommandDispatcher.dispatch("/compact")
    │ looks up registered handler
    ▼
compact_handler(args, context)
    │ performs microcompaction
    ▼
CommandResult(output="Compacted 5 messages", handled=True)
    │
    ▼
Middleware injects AIMessage with result
    │ short-circuits the LLM call
    ▼
Result streamed to client as command_result SSE event
```

## Key Properties

- **Transport-independent** — the dispatcher works with any message transport, not just HTTP
- **Short-circuit** — handled commands skip the LLM entirely
- **Extensible** — register custom commands via the dispatcher
- **Case-insensitive** — `/COMPACT` works the same as `/compact`
