# Echo Agent

**Source:** `src/langgraph_kit/graphs/echo_agent.py`
**Agent ID:** `echo-agent`

The minimal reference implementation demonstrating the standard agent contract.

## Architecture

```
START → llm_node → END
```

A single-node graph with:
- **State:** `MessagesState` (messages list with `add_messages` reducer)
- **Node:** `llm_node` — calls the LLM with the current messages
- **Flow:** Linear START → llm → END

## Implementation

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph_kit import build_llm


def build_graph(checkpointer, store):
    llm = build_llm()

    async def llm_node(state: MessagesState):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", llm_node)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)

    return graph.compile(checkpointer=checkpointer, store=store)
```

## Features

- No tools
- No memory
- No commands
- No middleware
- Basic conversation with message history (via checkpointer)

## Use Case

- Smoke testing the LLM connection
- Baseline for performance comparison
- Starting point for understanding the agent contract
- Simple chatbot without advanced features
