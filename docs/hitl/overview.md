# Human-in-the-Loop Overview

The HITL system enables agents to pause execution and request human approval before performing destructive or high-risk operations.

## Components

| Module | Purpose |
|--------|---------|
| [Models](models.md) | `ActionRequest`, `HumanInterrupt`, `HumanResponse`, `ResumeRequest` |
| [Tools](tools.md) | `approve_action` tool and interrupt mechanics |

## Flow

```
Agent decides to delete a file
    │
    ▼
Agent calls approve_action("delete_file", "Remove old config", {"path": "config.yaml"})
    │
    ▼
LangGraph interrupt() pauses the graph
    │
    ▼
SSE stream emits: {"interrupt": {"action_request": {...}, "config": {...}, "description": "..."}}
    │
    ▼
Frontend renders approval banner
    │
    ▼
User clicks "Accept" / "Reject" / types response
    │
    ▼
POST /agents/{id}/threads/{tid}/resume
    body: {"responses": [{"type": "accept"}]}
    │
    ▼
Graph resumes from interrupt point
    │
    ▼
approve_action returns "Approved" or raises rejection
```

## When to Use

HITL is appropriate for:
- **Destructive operations** — file deletion, database drops, branch deletion
- **External actions** — sending emails, posting to APIs, deploying code
- **High-stakes decisions** — anything that's hard to reverse

Tools with `interrupt_before=True` automatically pause for approval.
