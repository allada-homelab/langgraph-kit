# Empty Turn Middleware

**Source:** `src/langgraph_kit/core/resilience/empty_turn.py`

Detects and handles turns where the model produces no meaningful output.

## Class: EmptyTurnMiddleware

### Hook: aafter_model()

After each LLM response, checks if the response is effectively empty:
- No text content (or content shorter than 5 characters)
- No tool calls

### Nudge Injection

When an empty turn is detected, injects a nudge message asking the agent to take concrete action — either respond to the user or use a tool to make progress.

### Safety Limits

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_nudges` | 2 | Maximum consecutive nudges before giving up |
| `_MIN_CONTENT_LENGTH` | 5 | Minimum characters for meaningful content |

After `max_nudges` consecutive empty turns, the middleware stops intervening to prevent infinite loops.
