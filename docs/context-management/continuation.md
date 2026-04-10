# Continuation Tracker

**Source:** `src/langgraph_kit/core/context_management/continuation.py`

Token-budget continuation policy with diminishing-returns detection for long-running agent tasks.

## ContinuationDecision

```python
class ContinuationDecision(BaseModel):
    action: str                  # "continue" or "stop"
    reason: str                  # Human-readable explanation
    budget_consumed_pct: float   # Percentage of token budget used
    continuation_count: int      # How many continuations so far
    total_tokens_used: int       # Cumulative tokens
    diminishing_returns: bool    # Whether DR was detected
```

## Class: ContinuationTracker

### Constructor

```python
ContinuationTracker(
    budget_tokens: int = 100_000,
    max_continuations: int = 20,
    stop_threshold_pct: float = 0.90,
    diminishing_returns_ratio: float = 0.3,
    min_turns_for_dr: int = 3,
)
```

### Methods

#### record_turn(tokens_used: int)

Record the tokens used in the latest turn. Increments the continuation counter and updates cumulative totals.

#### should_continue() -> ContinuationDecision

Evaluate whether the agent should continue working:

1. **Max continuations** — if `continuation_count >= max_continuations`, stop
2. **Budget exhausted** — if `budget_consumed_pct >= stop_threshold_pct`, stop
3. **Diminishing returns** — if recent turns produce significantly less output than earlier turns, stop
4. Otherwise, continue

#### reset()

Reset the tracker for a new request.

### Diminishing Returns Detection

After at least `min_turns_for_dr` turns (default 3):

1. Split turn history into earlier turns and last 2 turns
2. Compute average tokens per turn for each group
3. If `recent_avg / earlier_avg < diminishing_returns_ratio` (default 0.3), DR is detected

This prevents the agent from spinning on a problem where each turn produces less useful output than the last.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget_tokens` | 100,000 | Total token budget for the request |
| `max_continuations` | 20 | Hard limit on loop iterations |
| `stop_threshold_pct` | 0.90 | Stop when 90% of budget consumed |
| `diminishing_returns_ratio` | 0.3 | Ratio below which DR is flagged |
| `min_turns_for_dr` | 3 | Minimum turns before DR detection kicks in |

## Usage

```python
tracker = ContinuationTracker(budget_tokens=50_000)

while True:
    tokens = await run_agent_turn()
    tracker.record_turn(tokens)

    decision = tracker.should_continue()
    if decision.action == "stop":
        print(f"Stopping: {decision.reason}")
        break
```
