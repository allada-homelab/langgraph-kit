# Observability

**Source:** `src/langgraph_kit/observability.py`

The observability module provides Langfuse integration for tracing agent invocations and a helper for building LangGraph run configs.

## UserInfo Protocol

```python
class UserInfo(Protocol):
    id: str | uuid.UUID
    email: str
```

A structural typing protocol that any user object can satisfy. Used to tag traces with user identity.

## API

### build_agent_run_config(thread_id, *, user=None, tags=None, metadata=None)

```python
def build_agent_run_config(
    thread_id: str,
    *,
    user: UserInfo | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]
```

Builds a LangGraph-compatible runnable config dict with:
- `configurable.thread_id` — conversation thread identifier
- `tags` — optional list of string tags
- `metadata` — optional metadata dict (includes `user_id` and `user_email` if user provided)
- `callbacks` — Langfuse callback handler (if tracing is enabled)

### langfuse_enabled()

```python
def langfuse_enabled() -> bool
```

Returns `True` if Langfuse tracing is fully configured (enabled flag + both keys set).

### init_langfuse()

```python
def init_langfuse() -> Langfuse | None
```

Initialize and return the shared Langfuse client singleton. Returns `None` if tracing is not configured. Subsequent calls return the same instance.

### create_langfuse_handler(*, user_id=None, session_id=None, tags=None, metadata=None)

```python
def create_langfuse_handler(...) -> CallbackHandler | None
```

Create a per-request Langfuse callback handler for LangChain. Returns `None` if tracing is not configured.

### flush_langfuse()

```python
def flush_langfuse() -> None
```

Flush pending Langfuse spans. Safe to call even if Langfuse is not initialized. Called automatically at the end of each stream.

### shutdown_langfuse()

```python
def shutdown_langfuse() -> None
```

Flush and close the Langfuse client. Call during application shutdown.

## Usage

```python
from langgraph_kit import build_agent_run_config, stream_agent_events

config = build_agent_run_config(
    thread_id="abc-123",
    user=current_user,
    tags=["production"],
)

async for chunk in stream_agent_events(graph, input_data, config):
    yield chunk
```

If Langfuse is configured, every LLM call, tool invocation, and agent step is automatically traced with user attribution and thread correlation.
