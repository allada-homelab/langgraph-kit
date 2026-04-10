# Async Tasks

**Source:** `src/langgraph_kit/core/orchestration/async_tasks.py`

Fire-and-forget background sub-agents with Store-backed tracking.

## AsyncTaskStatus

```python
class AsyncTaskStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
```

## AsyncTask

```python
class AsyncTask(BaseModel):
    task_id: str
    agent_name: str
    description: str
    thread_id: str                 # The sub-agent's thread
    status: AsyncTaskStatus
    created_at: str                # ISO timestamp
    last_checked_at: str = ""
    result: str = ""               # Output on success, error message on error

    def is_terminal(self) -> bool: ...
```

## Class: AsyncTaskManager

### Constructor

```python
AsyncTaskManager(
    store: BaseStore,
    parent_thread_id: str,
    available_graphs: dict[str, CompiledGraph],
)
```

### Methods

#### start(description, agent_name, input_messages) -> AsyncTask

Launch a background task:
1. Create a new thread ID for the sub-agent
2. Store task metadata in `("async_tasks", parent_thread_id, task_id)`
3. Start the graph execution in a background asyncio task
4. Return immediately with the task metadata

#### check(task_id) -> AsyncTask | None

Check the status and result of a task. Updates `last_checked_at`.

#### cancel(task_id) -> bool

Cancel a running task. Sets status to `CANCELLED`.

#### list_tasks(status_filter=None) -> list[AsyncTask]

List all tasks, optionally filtered by status.

## Agent-Callable Tools

`build_async_task_tools(manager, available_graphs)` returns 4 tools:

| Tool | Risk | Description |
|------|------|-------------|
| `start_async_task` | MUTATING | Launch a background task |
| `check_async_task` | READ_ONLY | Check task status |
| `cancel_async_task` | MUTATING | Cancel a running task |
| `list_async_tasks` | READ_ONLY | List all tasks |

## Store Namespace

```
("async_tasks", parent_thread_id, task_id) → AsyncTask.model_dump()
```

## Example

```python
manager = AsyncTaskManager(store, "main-thread", {"researcher": researcher_graph})

task = await manager.start(
    description="Find API documentation for auth endpoints",
    agent_name="researcher",
    input_messages=[HumanMessage("Find auth API docs")],
)

# Later...
result = await manager.check(task.task_id)
if result.status == AsyncTaskStatus.SUCCESS:
    print(result.result)
```
