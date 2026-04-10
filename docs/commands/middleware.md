# Command Middleware

**Source:** `src/langgraph_kit/core/commands/middleware.py`

Middleware that intercepts slash-commands from user messages before they reach the LLM.

## Class: CommandMiddleware

Extends `_AgentMiddleware`.

### Hook: abefore_agent()

Runs before the agent loop starts. Checks the last user message for a command prefix:

1. Extracts the last `HumanMessage` from the conversation
2. Calls `dispatcher.is_command(text)` to check for a match
3. If matched, calls `dispatcher.dispatch(text, context)` to execute
4. If `result.handled`:
   - Injects the command output as an `AIMessage`
   - Short-circuits the agent loop (no LLM call)
   - If metadata contains `compacted_messages`, replaces the message list

## Behavior

When a command is handled:
- The LLM is **never called** — the command result is the entire response
- The result appears as a `command_result` SSE event (not `token` events)
- The conversation state is updated with the command exchange

When a command is not recognized:
- The message passes through to the LLM as a normal user message
- The LLM sees the `/command` text and may respond to it naturally

## Integration

Part of the standard middleware stack, positioned first so commands are intercepted before any other processing:

```python
middleware_stack = [
    CommandMiddleware(dispatcher),  # First — intercept commands
    QueuedInputMiddleware(...),
    ToolErrorMiddleware(...),
    ...
]
```
