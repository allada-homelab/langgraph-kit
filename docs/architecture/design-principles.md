# Design Principles

## Composition Over Inheritance

Every subsystem (memory, tools, commands, prompts, resilience) is designed as an independent module with its own models and interfaces. Agent builders compose these pieces together rather than extending a monolithic base class.

## Convention-Based Registration

Agents, tools, commands, and skills all follow a register-then-lookup pattern:

```python
# Register once at startup
register("my-agent", graph)
tool_registry.register(capability)
dispatcher.register("/foo", handler)

# Lookup at runtime
graph = get("my-agent")
tools = tool_registry.compile_tools()
result = dispatcher.dispatch("/foo")
```

## Store as Universal Backend

All persistent data beyond conversation messages flows through LangGraph's `BaseStore` interface. This gives a single persistence backend (PostgreSQL in production, in-memory for testing) with hierarchical namespace-based organization.

## Stable-First Prompt Ordering

Prompt sections are classified by stability (STABLE, VOLATILE, CONDITIONAL). The composer places stable sections first to maximize Anthropic prompt cache hits — stable content rarely changes between turns, so it stays in the cache prefix.

## Sentinel-Based Streaming

Rich UI events (artifacts, progress indicators, suggestions) are emitted as tool outputs with sentinel prefixes (e.g., `__artifact__:{"type":"CODE",...}`). The streaming layer detects these sentinels and converts them to typed SSE events, keeping the streaming protocol simple while supporting rich interactions.

## Middleware Stack Pattern

Cross-cutting concerns (command interception, error recovery, memory extraction, context pressure) are implemented as middleware that wraps the agent loop. Each middleware has `abefore_agent`, `abefore_model`, `aafter_model`, and `awrap_tool_call` hooks, composing cleanly without modifying agent logic.

## Progressive Disclosure

Not every agent needs every feature. The echo agent uses none of the core subsystems. The reference deep agent uses everything. Subsystems are opt-in — you import and wire up only what you need.

## Provider Agnosticism

The LLM factory auto-detects the provider from the model name (`claude-*` → Anthropic, `gemini-*` → Google, default → OpenAI-compatible). All other code works with the LangChain `BaseChatModel` interface, never with provider-specific APIs directly.

## Frozen Configuration

`AgentConfig` is a frozen dataclass — immutable after creation. This prevents subtle bugs from runtime config mutations and makes the configuration deterministic from startup.
