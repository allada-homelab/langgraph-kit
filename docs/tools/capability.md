# Tool Capability

**Source:** `src/langgraph_kit/core/tools/capability.py`

## ToolRisk

```python
class ToolRisk(str, Enum):
    READ_ONLY = "read_only"       # No side effects
    MUTATING = "mutating"         # Modifies state
    DESTRUCTIVE = "destructive"   # Hard to reverse
```

## ToolCapability

```python
class ToolCapability(BaseModel):
    id: str                              # Unique identifier
    name: str                            # Display name
    description: str                     # Tool description (shown to LLM)
    fn: Callable                         # The actual callable
    tags: list[str] = []                 # Arbitrary labels for filtering
    risk: ToolRisk = ToolRisk.READ_ONLY  # Risk classification
    prompt_guidance: str = ""            # Injected into system prompt
    profiles: list[str] = []            # Applicable profiles (e.g., "coding")
    worker_types: list[str] = []        # Applicable worker types
    max_output_tokens: int = 0          # Hint for output size
    offload_large_results: bool = False  # Store large outputs externally
    interrupt_before: bool = False       # Pause for approval before execution
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | _(required)_ | Unique identifier for lookup and deduplication |
| `name` | `str` | _(required)_ | Human-readable name |
| `description` | `str` | _(required)_ | Description shown to the LLM for tool selection |
| `fn` | `Callable` | _(required)_ | The async function to execute |
| `tags` | `list[str]` | `[]` | Labels for filtering (e.g., `["search", "web"]`) |
| `risk` | `ToolRisk` | `READ_ONLY` | Risk level for filtering and approval logic |
| `prompt_guidance` | `str` | `""` | Prompt fragment injected when this tool is active |
| `profiles` | `list[str]` | `[]` | Profiles this tool applies to (empty = all) |
| `worker_types` | `list[str]` | `[]` | Worker types this tool applies to (empty = all) |
| `max_output_tokens` | `int` | `0` | Expected max output size (for context budgeting) |
| `offload_large_results` | `bool` | `False` | If true, large outputs are stored externally |
| `interrupt_before` | `bool` | `False` | If true, requires human approval before execution |

## Example

```python
from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

cap = ToolCapability(
    id="web-search",
    name="Web Search",
    description="Search the web for current information",
    fn=web_search_fn,
    tags=["search", "web"],
    risk=ToolRisk.READ_ONLY,
    prompt_guidance="Use web search for current events or real-time data.",
    profiles=["research", "general"],
    worker_types=["researcher"],
)
```
