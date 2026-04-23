# Basic Deep Agent

**Source:** `src/langgraph_kit/graphs/basic_deep_agent.py`
**Agent ID:** `basic-deep-agent`

The minimal deepagents example: framework defaults plus a generic system
prompt and a graph name for tracing. No kit features are wired in — use it
as a baseline to compare against the [reference deep agent](reference-deep-agent.md).

## Architecture

Uses `create_deep_agent()` from the deepagents package, which provides:

- Multi-step reasoning
- Tool use loops
- Conversation management

## Implementation

```python
from deepagents import create_deep_agent
from langgraph_kit import build_llm
from langgraph_kit.graphs._basic_prompt import BASIC_SYSTEM_PROMPT
from langgraph_kit.graphs._builder import DEFAULT_RECURSION_LIMIT


def build_basic_deep_agent(
    checkpointer, store, *, recursion_limit=DEFAULT_RECURSION_LIMIT
):
    graph = create_deep_agent(
        model=build_llm(),
        system_prompt=BASIC_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        name="basic-deep-agent",
    )
    return graph.with_config({"recursion_limit": recursion_limit})
```

## Recursion Limit

Defaults to `DEFAULT_RECURSION_LIMIT` (100). Override at build time with `recursion_limit=500`, or per-invocation with `config={"recursion_limit": 500}`. See the [agents overview](overview.md#recursion-limit) for details.

## Features

- deepagents framework features (reasoning, tool loops)
- Checkpointer-based persistence
- Store-based data access
- Generic "helpful assistant" system prompt shared with `echo-agent`
- Named graph (visible in LangGraph Studio / tracing)

## When to Use

Use the basic deep agent as a baseline when you want deepagents framework
features without the full kit stack. It's simpler than the reference agent
but more capable than the echo agent. Once you need persistent memory,
custom tools, slash commands, or multi-agent orchestration, move up to
`reference-deep-agent`.

## Dependencies

Requires the `deepagents` package (`>=0.4`), which is a core dependency of langgraph-kit.
