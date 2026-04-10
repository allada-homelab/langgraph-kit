# Pressure Monitor

**Source:** `src/langgraph_kit/core/context_management/pressure.py`

## MitigationStrategy

```python
class MitigationStrategy(Enum):
    NONE = "none"
    MICROCOMPACT = "microcompact"
    SESSION_ASSISTED = "session_assisted"
    FULL_COMPACTION = "full_compaction"
    STOP = "stop"
```

## PressureSignals

```python
class PressureSignals(BaseModel):
    estimated_tokens: int         # Total estimated tokens in conversation
    window_limit: int             # Configured token window size
    pressure_pct: float           # estimated_tokens / window_limit
    large_tool_outputs: int       # Count of large tool outputs
    compaction_failures: int      # Consecutive compaction failures
```

## Class: PressureMonitor

### Constructor

```python
PressureMonitor(
    window_limit: int = 128_000,
    warn_pct: float = 0.70,
    critical_pct: float = 0.85,
    max_compaction_failures: int = 3,
    large_output_threshold: int = 4000,  # tokens
)
```

### Methods

#### assess(messages) -> PressureSignals

Calculate pressure signals from the current message list.

- Estimates tokens using 4 chars ≈ 1 token heuristic
- Counts tool outputs exceeding `large_output_threshold` tokens

#### choose_mitigation(signals) -> MitigationStrategy

Select a mitigation strategy based on pressure signals:

| Condition | Strategy |
|-----------|----------|
| `compaction_failures >= max_compaction_failures` | `STOP` |
| `pressure_pct < warn_pct` | `NONE` |
| `warn_pct <= pressure_pct < critical_pct` | `MICROCOMPACT` |
| `pressure_pct >= critical_pct` | `FULL_COMPACTION` |

#### record_compaction_failure()

Increment the failure counter. After `max_compaction_failures`, the strategy switches to `STOP`.

#### record_compaction_success()

Reset the failure counter to 0.

#### reset()

Clear all tracked state.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_limit` | 128,000 | Token window size (model context length) |
| `warn_pct` | 0.70 | Pressure threshold for microcompaction |
| `critical_pct` | 0.85 | Pressure threshold for full compaction |
| `max_compaction_failures` | 3 | Circuit breaker limit |
| `large_output_threshold` | 4,000 | Tokens; outputs larger than this are counted |
