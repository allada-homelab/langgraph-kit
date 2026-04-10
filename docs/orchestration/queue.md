# Message Queue

**Source:** `src/langgraph_kit/core/orchestration/queue.py`

Store-backed per-thread message queue for buffering user input while an agent is busy.

## QueueSemantic

```python
class QueueSemantic(str, Enum):
    APPEND = "append"             # Add to end of queue
    INTERRUPT = "interrupt"       # Signal urgency
    REPLACE_GOAL = "replace_goal" # Clear queue and replace with this
```

## QueuedItem

```python
class QueuedItem(BaseModel):
    id: str                        # UUID
    content: str                   # Message content
    semantic: QueueSemantic        # How to handle this item
    source: str = "user"           # Origin
    metadata: dict = {}            # Arbitrary metadata
    timestamp: str                 # ISO timestamp
```

## ThreadQueue

### Constructor

```python
ThreadQueue(store: BaseStore, thread_id: str)
```

Stored at namespace `("queue", thread_id)`.

### Methods

| Method | Description |
|--------|-------------|
| `enqueue(content, semantic, source, metadata)` | Add item (replaces all if `REPLACE_GOAL`) |
| `drain()` | Remove and return all items in FIFO order |
| `peek()` | View items without removing |
| `depth()` | Count items in queue |
| `clear()` | Remove all items |

## ThreadBusyTracker

Tracks which threads have active agent runs.

### Constructor

```python
ThreadBusyTracker(store: BaseStore)
```

Stored at namespace `("thread_busy",)`.

### Methods

| Method | Description |
|--------|-------------|
| `mark_busy(thread_id)` | Mark thread as busy |
| `mark_idle(thread_id)` | Mark thread as idle |
| `is_busy(thread_id)` | Check if thread is busy |

Auto-expires after 600 seconds to prevent stale locks from crashed processes.

## QueuedInputMiddleware

### Hook: abefore_model()

Before each LLM call, drains the queue and injects queued messages as `HumanMessage` instances into the conversation state.

## HTTP Endpoints

The FastAPI integration exposes queue operations:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agents/{id}/threads/{tid}/queue` | Enqueue a message |
| `GET` | `/agents/{id}/threads/{tid}/queue` | Check queue status |

## Flow

```
1. User sends message → POST /stream starts agent run
2. Agent run marks thread as busy (ThreadBusyTracker)
3. User sends another message → POST /queue buffers it
4. Agent turn ends → QueuedInputMiddleware drains queue
5. Agent processes queued messages in next turn
6. Agent run completes → thread marked idle
```
