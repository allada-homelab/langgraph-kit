# Context Providers

**Source:** `src/langgraph_kit/core/prompt_assembly/context_providers.py`

Dynamic context providers inject runtime information into the system prompt during composition.

## ContextProvider Protocol

```python
class ContextProvider(Protocol):
    async def provide(self, context: dict) -> str:
        """Return a string to append to the system prompt."""
        ...
```

Any object implementing this protocol can be passed to `PromptComposer`.

## Built-in Providers

### ThreadContextProvider

Injects thread metadata:
- Thread ID
- Message count
- Other thread-specific state

### MemoryContextProvider

Injects relevant memory summaries:
- Loads memories from the configured scope
- Formats as a readable list for the system prompt
- Enables the agent to leverage stored knowledge without explicit search

### ToolContextProvider

Injects tool usage guidance:
- Collects `prompt_guidance` fragments from all registered tools
- Formats as a guidance section in the prompt
- Helps the agent understand when and how to use available tools

### GitContextProvider

**Source:** `src/langgraph_kit/core/prompt_assembly/git_context.py`

Injects git repository context (used by the coding agent):
- Current branch name
- Recent commit messages
- Working tree status (modified/staged files)
- Repository root path

## Custom Providers

```python
class MyProvider:
    async def provide(self, context: dict) -> str:
        thread_id = context.get("thread_id", "")
        return f"Current thread: {thread_id}\nCustom context here..."

composer = PromptComposer(
    section_registry=registry,
    context_providers=[MyProvider()],
)
```

## Provider vs Section

| Aspect | PromptSection | ContextProvider |
|--------|---------------|-----------------|
| Content | Static text | Dynamic (computed at runtime) |
| Caching | Cached by stability | Never cached |
| Ordering | Sorted by stability + priority | Appended after all sections |
| Use case | Identity, instructions, workflows | Thread state, live data, memory |
