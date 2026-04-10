# Builder Utilities (Deprecated)

> **This module has been refactored.** The builder utilities have moved from `graphs/_builder.py` to `core/graph_builder/`. See [Graph Builder](graph-builder.md) for the current documentation.

`graphs/_builder.py` still exists as a backwards-compatible re-export shim. New code should import from `langgraph_kit.core.graph_builder`:

```python
# Old (still works)
from langgraph_kit.graphs._builder import build_middleware_stack

# New (preferred)
from langgraph_kit.core.graph_builder import build_middleware_stack
```
