# Completion Guard

**Source:** `src/langgraph_kit/core/resilience/completion_guard.py`

Heuristic detector for premature agent completion — when the model claims to be done but likely has unfinished work.

## Class: CompletionGuardMiddleware

### Hook: aafter_model()

After each LLM response, checks for suspicious completion signals:

1. **Recent tool error unrecovered** — a tool failed but the agent declared success without retrying
2. **No tools used despite context** — the agent has tools available but never used them
3. **Minimal exploration** — the agent stopped after very few steps

### Detection

The guard analyzes the last 20 messages for patterns:
- Completion phrases matching `_COMPLETION_PHRASES` regex (e.g., "I've completed", "All done", "The task is finished")
- Combined with situational signals (recent errors, low tool usage)

### Challenge Injection

When a suspicious completion is detected, the guard injects a challenge message asking the agent to either:
- Justify that the work is truly complete
- Continue with remaining steps

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `_MIN_MESSAGES_BEFORE_GUARD` | 4 | Let agent warm up before guarding |
| `_MAX_CHALLENGES` | 2 | Prevent infinite challenge loops |

After `_MAX_CHALLENGES`, the guard stops intervening and lets the agent complete.
