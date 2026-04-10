# Prompt Composer

**Source:** `src/langgraph_kit/core/prompt_assembly/composer.py`

Assembles the final system prompt from registered sections and dynamic context providers.

## Class: PromptComposer

### Constructor

```python
PromptComposer(
    section_registry: SectionRegistry,
    context_providers: list[ContextProvider] = [],
)
```

### Methods

#### compose(context: dict = {}) -> str

Assemble the full system prompt:

1. Get active sections from the registry (stable-first ordering)
2. Check the `PromptCache` for cached stable content
3. Append volatile and conditional section content
4. Run each `ContextProvider` with the context dict
5. Append provider outputs
6. Join everything with double newlines

#### compose_sections_only() -> str

Assemble prompt from sections only, without running context providers. Useful for testing or when dynamic context is not needed.

#### get_active_section_ids() -> list[str]

Return the IDs of currently active sections (respecting conditions).

## Caching

The composer uses a `PromptCache` that stores the concatenated content of all stable sections. Since stable sections rarely change, this avoids recomputing the same text every turn. The cache key is derived from the combined `cache_key` fields of all stable sections.

## Example

```python
from langgraph_kit.core.prompt_assembly.composer import PromptComposer
from langgraph_kit.core.prompt_assembly.sections import SectionRegistry
from langgraph_kit.core.prompt_assembly.context_providers import MemoryContextProvider

registry = SectionRegistry()
# ... register sections ...

composer = PromptComposer(
    section_registry=registry,
    context_providers=[
        MemoryContextProvider(memory_mgr),
        ToolContextProvider(tool_registry),
    ],
)

system_prompt = await composer.compose({"thread_id": "abc-123"})
```
