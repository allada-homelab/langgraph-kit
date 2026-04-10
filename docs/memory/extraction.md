# Auto-Extraction

**Source:** `src/langgraph_kit/core/memory/extraction.py`

The `AutoMemoryExtractor` identifies and persists durable facts from recent conversation turns without explicit user action.

## Class: AutoMemoryExtractor

### Constructor

```python
AutoMemoryExtractor(
    memory_manager: PersistentMemoryManager,
    llm: BaseChatModel,
)
```

### Methods

#### extract(messages, scope=MemoryScope.ASSISTANT) -> list[MemoryRecord]

Analyze recent messages and create/update/delete memory records as appropriate.

**Behavior:**
1. Formats existing memories for the LLM to see what's already stored
2. Formats recent messages as conversation context
3. Asks the LLM to identify durable facts worth persisting
4. Parses the LLM's JSON response into create/update/delete actions
5. Executes the actions via `PersistentMemoryManager`

**Skip conditions:**
- If the agent already called memory tools this turn (avoids duplication)

## Extraction Prompt

The LLM is instructed to save only **future-useful facts** and avoid:
- Code patterns derivable from reading the repo
- Git history or recent changes
- Debugging solutions (the fix is in the code)
- Ephemeral task details
- Temporary conversation state

The response format is a JSON array of actions:
```json
[
    {"action": "create", "title": "...", "type": "user", "summary": "...", "body": "..."},
    {"action": "update", "id": "...", "body": "updated content"},
    {"action": "delete", "id": "..."}
]
```

## ExtractionMiddleware

The extraction runs as post-turn middleware via `ExtractionMiddleware`, which calls `extract()` after each agent response. This is part of the standard middleware stack built by `build_middleware_stack()`.

## Example Flow

```
User: "I'm a data scientist investigating logging"
    │
Agent responds with logging information
    │
ExtractionMiddleware (post-turn):
    │
    ├── Loads existing memories
    ├── Formats recent messages
    ├── LLM identifies: user is a data scientist, focused on observability
    └── Creates MemoryRecord(type=USER, title="Data scientist role", ...)
```
