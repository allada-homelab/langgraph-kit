# UI Events & Artifacts Overview

The UI events system enables agents to emit rich, typed events alongside streaming text tokens. These events drive frontend components like code blocks, progress bars, suggestion chips, and citation cards.

## Components

| Module | Purpose |
|--------|---------|
| [Artifacts](artifacts.md) | Structured code, markdown, diagrams, tables |
| [Progress & Suggestions](progress-suggestions.md) | Ephemeral UI events |

## Sentinel Mechanism

UI events are emitted as tool outputs with sentinel prefixes. The streaming layer detects these sentinels and converts them to typed SSE events:

```
Tool output:  __artifact__:{"type":"CODE","title":"main.py","content":"..."}
     ↓
SSE event:    data: {"artifact": {"type": "CODE", "title": "main.py", "content": "..."}}
```

| Sentinel | SSE Key | Tool |
|----------|---------|------|
| `__artifact__:` | `artifact` | `create_artifact` |
| `__progress__:` | `progress` | `emit_progress` |
| `__suggestions__:` | `suggestions` | `suggest_actions` |
| `__citation__:` | `citation` | `add_citation` |

## Why Sentinels?

This approach keeps the streaming protocol simple — tool outputs are just strings. The streaming layer handles the conversion to typed events without needing special protocol extensions. The sentinel prefix is stripped before parsing the JSON payload.
