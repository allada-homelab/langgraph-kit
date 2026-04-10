# FastAPI Integration

**Source:** `src/langgraph_kit/contrib/fastapi.py`
**Extra required:** `langgraph-kit[fastapi]`

A route factory that creates a FastAPI `APIRouter` with 11 endpoints for agent interaction.

## create_agent_router(get_current_user)

```python
def create_agent_router(
    get_current_user: Callable,  # FastAPI dependency returning UserInfo
) -> APIRouter
```

Returns an `APIRouter` with all agent endpoints. The `get_current_user` parameter is a FastAPI dependency that returns an object satisfying the `UserInfo` protocol (must have `id` and `email` attributes).

## Endpoints

### Agent Discovery

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/agents/` | List all registered agents | `AgentListResponse` |

### Conversation

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/agents/{agent_id}/stream` | Stream tokens as SSE | `text/event-stream` |
| `POST` | `/agents/{agent_id}/invoke` | Full response (JSON) | `InvokeResponse` |

### Thread State

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/agents/{agent_id}/threads/{thread_id}/messages` | Load conversation history | `list[ChatMessage]` |
| `GET` | `/agents/{agent_id}/threads/{thread_id}/state` | Check for interrupts | `ThreadStateResponse` |

### Message Queue

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/agents/{agent_id}/threads/{thread_id}/queue` | Enqueue message | `QueueMessageResponse` |
| `GET` | `/agents/{agent_id}/threads/{thread_id}/queue` | Check queue status | `QueueStatusResponse` |

### Human-in-the-Loop

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/agents/{agent_id}/threads/{thread_id}/resume` | Resume interrupted thread | `InvokeResponse` |
| `POST` | `/agents/{agent_id}/threads/{thread_id}/resume/stream` | Resume with streaming | `text/event-stream` |

### Branching

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/agents/{agent_id}/threads/{thread_id}/history` | Checkpoint history | `list[CheckpointInfo]` |
| `POST` | `/agents/{agent_id}/threads/{thread_id}/fork` | Fork at checkpoint | `ForkResponse` |

## Request/Response Models

### InvokeRequest

```python
class InvokeRequest(BaseModel):
    messages: list[ChatMessage]  # Conversation messages
    thread_id: str = ""          # Thread ID (auto-generated if empty)
    checkpoint_id: str = ""      # For branching: start from this checkpoint
```

### ChatMessage

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
```

### InvokeResponse

```python
class InvokeResponse(BaseModel):
    content: str      # Full response text
    thread_id: str    # Thread ID used
```

## Command Dispatch

The `/stream` and `/invoke` endpoints check if the last user message is a slash command. If it matches a registered command, the command is dispatched directly without calling the LLM.

## Setup Example

```python
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from langgraph_kit import AgentConfig, configure, create_persistence
from langgraph_kit.contrib.fastapi import create_agent_router
from langgraph_kit.graphs import register_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure(AgentConfig(llm_model="gpt-4o", llm_api_key="sk-..."))
    async with create_persistence() as (checkpointer, store):
        await register_all(checkpointer, store, mcp_tools=[])
        app.state.store = store
        yield

app = FastAPI(lifespan=lifespan)

# Your auth dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    return verify_token(token)

router = create_agent_router(get_current_user=get_current_user)
app.include_router(router, prefix="/api/v1")
```

## SSE Streaming Format

The `/stream` endpoint returns `text/event-stream` with events as described in [SSE Event Types](../api-reference/sse-events.md). Each event is a `data:` line followed by a JSON payload and two newlines.

## Store Access

The router accesses the LangGraph Store via `request.app.state.store`. This must be set during application startup (typically in the lifespan handler).
