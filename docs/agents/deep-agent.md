# Deep Agent

**Source:** `src/langgraph_kit/graphs/deep_agent.py`
**Agent ID:** `deep-agent`

A baseline agent using the [deepagents](https://pypi.org/project/deepagents/) framework.

## Architecture

Uses `create_deep_agent()` from the deepagents package, which provides:
- Multi-step reasoning
- Tool use loops
- Conversation management

## Implementation

```python
from deepagents import create_deep_agent
from langgraph_kit import build_llm


def build_deep_graph(checkpointer, store):
    llm = build_llm()
    return create_deep_agent(llm, checkpointer=checkpointer, store=store)
```

## Features

- deepagents framework features (reasoning, tool loops)
- Checkpointer-based persistence
- Store-based data access

## When to Use

Use the deep agent as a baseline when you want deepagents framework features without the full R0/coding stack. It's simpler than R0 but more capable than the echo agent.

## Dependencies

Requires the `deepagents` package (`>=0.4`), which is a core dependency of langgraph-kit.
