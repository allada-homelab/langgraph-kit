# Plugin Loader

**Source:** `src/langgraph_kit/core/plugins/loader.py` and `src/langgraph_kit/core/plugins/registry.py`

Load Python plugin files that extend agent capabilities.

## Plugin Contract

A plugin is a Python file with a `contribute()` function:

```python
# my_plugin.py
from langgraph_kit.core.plugins.registry import PluginContribution

def contribute() -> PluginContribution:
    return PluginContribution(
        tools=[my_tool_capability],
        sections=[my_prompt_section],
    )
```

## PluginContribution

```python
class PluginContribution:
    tools: list[ToolCapability] = []
    sections: list[PromptSection] = []
```

## PluginLoader

### Methods

| Method | Description |
|--------|-------------|
| `load_from_directory(path)` | Scan directory for `.py` files with `contribute()` |
| `get_contributions()` | Return all collected contributions |

## PluginRegistry

Aggregates contributions from all plugins:

### Methods

| Method | Description |
|--------|-------------|
| `contrib()` | Return combined `PluginContribution` from all loaded plugins |

## Configuration

Set the `plugins_dir` field in `AgentConfig`:

```python
AgentConfig(plugins_dir="/app/plugins/")
```

During agent startup, the plugin loader scans this directory and calls `contribute()` on each discovered plugin file.
