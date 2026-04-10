# SSE Streaming

**Source:** `src/langgraph_kit/streaming.py`

The streaming module converts LangGraph's `astream_events` v2 into Server-Sent Events (SSE) suitable for real-time frontend consumption.

## API

### stream_agent_events(graph, input_data, config, *, store=None)

```python
async def stream_agent_events(
    graph: Any,
    input_data: dict[str, Any],
    config: dict[str, Any],
    *,
    store: Any | None = None,
) -> AsyncGenerator[str]:
```

Returns an async generator of SSE-formatted strings (`data: {...}\n\n`).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `CompiledGraph` | The LangGraph graph to execute |
| `input_data` | `dict` | Input data (typically `{"messages": [...]}`) |
| `config` | `dict` | LangGraph config (must include `configurable.thread_id`) |
| `store` | `BaseStore` | Optional store for busy-thread tracking |

## Event Types

| Event | Format | Description |
|-------|--------|-------------|
| Token | `{"token": "..."}` | Text token from the LLM |
| Tool call start | `{"tool_call_start": {"id": "...", "name": "...", "args": {...}}}` | Tool invocation began |
| Tool call end | `{"tool_call_end": {"id": "...", "name": "...", "output": "..."}}` | Tool completed |
| Artifact | `{"artifact": {...}}` | Structured UI artifact (code, diagram, etc.) |
| Progress | `{"progress": {"step": "...", "current": N, "total": M}}` | Progress indicator |
| Suggestions | `{"suggestions": {"actions": [...]}}` | Suggested follow-up actions |
| Citation | `{"citation": {"title": "...", "source": "...", "snippet": "..."}}` | Source citation |
| Command result | `{"command_result": {"output": "..."}}` | Slash-command output (no LLM tokens) |
| Interrupt | `{"interrupt": {...}}` | Graph paused for human input |
| Done | `[DONE]` | Stream finished |

See [SSE Event Types](../api-reference/sse-events.md) for full format details.

## Sentinel Detection

Rich UI events (artifacts, progress, suggestions, citations) are emitted by tools as sentinel-prefixed strings:

```
__artifact__:{"type": "CODE", "title": "example.py", ...}
__progress__:{"step": "Searching", "current": 1, "total": 5}
__suggestions__:{"actions": ["Run tests", "Deploy"]}
__citation__:{"title": "RFC 7231", "source": "...", "snippet": "..."}
```

The streaming layer detects these prefixes and converts them to typed SSE events. Normal tool outputs are emitted as `tool_call_end` events.

## Text Buffering

Some models (notably Qwen via OpenAI-compatible APIs) leak the `tool_calls` list into text content as trailing `[]` or `` ```json\n[]\n``` ``. The streamer buffers the last 30 characters and strips these artifacts before sending.

## Tool Output Truncation

Tool outputs longer than 3,000 characters are truncated in the SSE stream with `...(truncated)`. The full output remains in the conversation state for the LLM.

## Busy Thread Tracking

When `store` is provided, the stream marks the thread as busy via `ThreadBusyTracker` for the duration of the run. This enables the queue endpoints to detect active threads and buffer messages.

## Langfuse Flush

On stream completion (in the `finally` block), `flush_langfuse()` is called to ensure all telemetry spans are sent before the connection closes.
