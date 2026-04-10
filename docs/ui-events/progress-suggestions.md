# Progress & Suggestions

**Source:** `src/langgraph_kit/core/ui_events.py`

Ephemeral UI events for progress indicators, action suggestions, and citations.

## emit_progress

Built by `build_progress_tool()`:

```python
async def emit_progress(step: str, current: int, total: int) -> str
```

Emits a progress indicator. The frontend can render this as a progress bar or step counter.

**SSE format:**
```json
{"progress": {"step": "Searching files", "current": 3, "total": 10}}
```

## suggest_actions

Built by `build_suggestions_tool()`:

```python
async def suggest_actions(actions: list[str]) -> str
```

Suggests follow-up actions the user might want to take. The frontend renders these as clickable chips or buttons.

**SSE format:**
```json
{"suggestions": {"actions": ["Run tests", "Deploy to staging", "Review PR"]}}
```

## add_citation

Built by `build_citation_tool()`:

```python
async def add_citation(title: str, source: str, snippet: str) -> str
```

Adds a source citation. The frontend can render these as reference cards.

**SSE format:**
```json
{"citation": {"title": "RFC 7231", "source": "https://...", "snippet": "The 200 status code..."}}
```

## Registration

All UI tools are registered via the builder utility:

```python
from langgraph_kit.core.graph_builder.tools import register_ui_tools

register_ui_tools(tool_registry)
# Registers: create_artifact, emit_progress, suggest_actions, add_citation
# All with ToolRisk.READ_ONLY
```
