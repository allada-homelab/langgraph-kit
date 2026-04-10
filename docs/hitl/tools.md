# HITL Tools

**Source:** `src/langgraph_kit/core/hitl/tools.py`

## build_approve_action_tool()

Returns an async tool function:

### approve_action(action, description, args)

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `str` | What the agent wants to do |
| `description` | `str` | Human-readable explanation for the user |
| `args` | `dict` | Action arguments to show the user |

**Behavior:**

1. Constructs a `HumanInterrupt` payload
2. Calls LangGraph's `interrupt()` to pause the graph
3. The interrupt payload is emitted as an SSE event
4. When the user resumes, the response is returned to the tool
5. Returns `"Approved"` on accept, or raises/returns the rejection

**Registration:**

```python
from langgraph_kit.core.graph_builder.tools import register_hitl_tools

register_hitl_tools(tool_registry)
# Registers approve_action with ToolRisk.READ_ONLY
# (the tool itself doesn't modify anything — it just pauses)
```

## Automatic Interrupts

Tools with `interrupt_before=True` in their `ToolCapability` will automatically interrupt before execution without needing to call `approve_action` explicitly. This is handled at the graph level by LangGraph's built-in interrupt mechanism.
