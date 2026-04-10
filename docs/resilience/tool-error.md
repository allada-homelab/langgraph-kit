# Tool Error Middleware

**Source:** `src/langgraph_kit/core/resilience/tool_error.py`

Catches tool exceptions and converts them to structured error results the agent can reason about.

## Class: ToolErrorMiddleware

### Hook: awrap_tool_call()

Wraps every tool invocation:

1. Execute the tool function
2. If it succeeds, return the normal result
3. If it raises an exception:
   - Check if the error is transient (retryable)
   - If retryable and under `max_retries`, retry the call
   - Convert the exception to a structured `ToolMessage` error

### Structured Error Format

```json
{
    "error_type": "ConnectionError",
    "detail": "Connection refused: localhost:5432",
    "retryable": true,
    "retries_exhausted": false
}
```

This gives the agent enough information to decide whether to retry, try a different approach, or report the error to the user.

### Transient Error Detection

The middleware recognizes these exception types as transient (retryable):

- Timeout errors
- Connection errors
- Rate limit errors

All other errors are treated as non-transient.

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_retries` | 1 | Number of retry attempts for transient errors |
