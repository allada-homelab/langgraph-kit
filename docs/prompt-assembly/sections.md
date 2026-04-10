# Prompt Sections

**Source:** `src/langgraph_kit/core/prompt_assembly/sections.py`

## SectionStability

```python
class SectionStability(str, Enum):
    STABLE = "stable"           # Rarely changes (cached)
    VOLATILE = "volatile"       # Changes every turn
    CONDITIONAL = "conditional" # Active based on predicate
```

## PromptSection

```python
class PromptSection(BaseModel):
    id: str                                     # Unique identifier
    content: str                                # The prompt text
    stability: SectionStability                 # Cache classification
    priority: int = 0                           # Sort order (lower = higher priority)
    condition: Callable[[], bool] | None = None # Predicate for CONDITIONAL sections
    cache_key: str = ""                         # Auto-generated from content hash
```

The `cache_key` is automatically generated from the content hash if not provided, enabling the prompt cache to detect changes.

## SectionRegistry

### Methods

| Method | Description |
|--------|-------------|
| `register(section)` | Add a section |
| `register_many(sections)` | Add multiple sections |
| `get(id)` | Retrieve by ID |
| `remove(id)` | Remove by ID |
| `get_active()` | Return sections filtered by stability and conditions, sorted by priority |
| `get_by_stability(stability)` | Return sections of specific stability |

### get_active() Behavior

1. Includes all `STABLE` sections
2. Includes all `VOLATILE` sections
3. Includes `CONDITIONAL` sections where `condition()` returns `True`
4. Sorts by `stability` (STABLE first), then by `priority`

## Example

```python
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection, SectionStability, SectionRegistry
)

registry = SectionRegistry()

registry.register(PromptSection(
    id="core_identity",
    content="You are a helpful coding assistant...",
    stability=SectionStability.STABLE,
    priority=0,
))

registry.register(PromptSection(
    id="coding_workflow",
    content="When writing code, follow these patterns...",
    stability=SectionStability.CONDITIONAL,
    priority=10,
    condition=lambda: profile == "coding",
))

active = registry.get_active()
# Returns core_identity always; coding_workflow only when condition is True
```
