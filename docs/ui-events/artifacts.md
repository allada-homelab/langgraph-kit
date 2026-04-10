# Artifacts

**Source:** `src/langgraph_kit/core/artifacts.py`

Structured UI artifacts for rich content rendering in the frontend.

## ArtifactType

```python
class ArtifactType(str, Enum):
    CODE = "code"           # Source code with syntax highlighting
    MARKDOWN = "markdown"   # Rendered markdown
    TABLE = "table"         # Tabular data
    DIAGRAM = "diagram"     # Mermaid or similar diagrams
    JSON = "json"           # Formatted JSON
    DIFF = "diff"           # Code diff
    HTML = "html"           # Raw HTML
```

## ArtifactEvent

```python
class ArtifactEvent(BaseModel):
    id: str                          # Unique artifact ID
    type: ArtifactType               # Artifact type
    title: str                       # Display title
    content: str                     # The artifact content
    language: str = ""               # Language for syntax highlighting
    metadata: dict[str, Any] = {}    # Additional metadata
```

## Tool: create_artifact

Built by `build_artifact_tool()`. Returns a function with signature:

```python
async def create_artifact(
    type: str,       # ArtifactType value
    title: str,      # Display title
    content: str,    # Content body
    language: str = "",  # Optional language
) -> str
```

Returns a sentinel-prefixed JSON string that the streaming layer converts to an SSE artifact event.

## Artifact Queue

For cases where artifacts need to be collected and processed in batch:

| Function | Description |
|----------|-------------|
| `init_artifact_queue()` | Reset the queue for the current context |
| `queue_artifact(event)` | Add an artifact to the queue |
| `drain_artifact_queue()` | Return and clear all queued artifacts |

## Example

When the agent creates an artifact:

```
Agent calls: create_artifact(type="code", title="auth.py", content="def login(): ...", language="python")
    ↓
Tool returns: __artifact__:{"id":"abc","type":"code","title":"auth.py","content":"def login(): ...","language":"python"}
    ↓
Streaming layer emits: data: {"artifact": {"id":"abc","type":"code",...}}
    ↓
Frontend renders: Syntax-highlighted code block titled "auth.py"
```
