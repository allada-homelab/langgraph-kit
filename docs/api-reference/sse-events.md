# SSE Event Types

Complete reference for all Server-Sent Events emitted by `stream_agent_events()`.

## Format

Each event is a `data:` line followed by a JSON object and two newlines:

```
data: {"token": "Hello"}\n\n
data: {"token": " world"}\n\n
data: {"tool_call_start": {"id": "abc", "name": "search", "args": {"q": "test"}}}\n\n
data: [DONE]\n\n
```

## Event Types

### token

LLM text output, streamed incrementally.

```json
{"token": "Here is the "}
{"token": "answer to your question."}
```

**Notes:**
- Leading whitespace is stripped from the first token
- Trailing model artifacts (e.g., Qwen `[]` leak) are stripped
- Text is buffered (30 chars) to enable artifact stripping

### tool_call_start

A tool invocation has begun.

```json
{
    "tool_call_start": {
        "id": "run-abc123",
        "name": "search_memories",
        "args": {"query": "user preferences", "scope": "user"}
    }
}
```

### tool_call_end

A tool invocation has completed.

```json
{
    "tool_call_end": {
        "id": "run-abc123",
        "name": "search_memories",
        "output": "Found 3 memories: ..."
    }
}
```

**Notes:**
- Outputs longer than 3,000 characters are truncated with `...(truncated)`
- Sentinel-prefixed outputs are emitted as their respective event types instead

### artifact

A structured UI artifact created via the `create_artifact` tool.

```json
{
    "artifact": {
        "id": "art-001",
        "type": "code",
        "title": "auth.py",
        "content": "def login(username, password): ...",
        "language": "python",
        "metadata": {}
    }
}
```

**Artifact types:** `code`, `markdown`, `table`, `diagram`, `json`, `diff`, `html`

### progress

A progress indicator from the `emit_progress` tool.

```json
{
    "progress": {
        "step": "Searching files",
        "current": 3,
        "total": 10
    }
}
```

### suggestions

Suggested follow-up actions from the `suggest_actions` tool.

```json
{
    "suggestions": {
        "actions": ["Run tests", "Deploy to staging", "Review PR"]
    }
}
```

### citation

A source citation from the `add_citation` tool.

```json
{
    "citation": {
        "title": "RFC 7231",
        "source": "https://tools.ietf.org/html/rfc7231",
        "snippet": "The 200 (OK) status code indicates..."
    }
}
```

### command_result

Output from a slash-command (emitted when no LLM tokens were streamed).

```json
{
    "command_result": {
        "output": "Context: 45,000 tokens (35% of 128K window)"
    }
}
```

### interrupt

The graph has paused for human input (HITL).

```json
{
    "interrupt": {
        "action_request": {
            "action": "delete_file",
            "args": {"path": "config.yaml"}
        },
        "config": {
            "allow_accept": true,
            "allow_ignore": true,
            "allow_respond": true,
            "allow_edit": true
        },
        "description": "Delete the old configuration file"
    }
}
```

### [DONE]

Stream finished. Always the last event.

```
data: [DONE]\n\n
```

## Frontend Consumption

```javascript
const eventSource = new EventSource('/api/v1/agents/reference-deep-agent/stream');

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }

    const data = JSON.parse(event.data);

    if (data.token) appendText(data.token);
    if (data.artifact) renderArtifact(data.artifact);
    if (data.progress) updateProgressBar(data.progress);
    if (data.suggestions) showSuggestionChips(data.suggestions);
    if (data.tool_call_start) showToolSpinner(data.tool_call_start);
    if (data.tool_call_end) hideToolSpinner(data.tool_call_end);
    if (data.interrupt) showApprovalBanner(data.interrupt);
    if (data.command_result) showCommandOutput(data.command_result);
};
```
