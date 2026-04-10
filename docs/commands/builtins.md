# Built-in Commands

**Source:** `src/langgraph_kit/core/commands/builtins.py`

Factory functions that create handlers for built-in slash commands. Each returns an async handler function suitable for `CommandDispatcher.register()`.

## /help

```
/help
```

Lists all registered commands with descriptions and usage. Built from the dispatcher's own command registry.

**Factory:** `build_help_command(dispatcher) -> CommandHandler`

## /memory

```
/memory [scope]
```

Inspects persistent memory records. If a scope is provided, lists records in that scope. Otherwise, lists all scopes with record counts.

**Factory:** `build_memory_command(memory_mgr) -> CommandHandler`

## /context

```
/context
```

Displays context window status: estimated tokens, pressure percentage, window limit, number of messages, and number of large tool outputs.

**Factory:** `build_context_command(pressure_monitor) -> CommandHandler`

## /compact

```
/compact
```

Performs **microcompaction** — truncates large tool outputs in older messages to reduce context pressure without losing conversational flow.

**Behavior:**
- Skips the most recent 10 messages (`_COMPACT_RECENT_WINDOW`)
- Truncates tool outputs larger than 2,000 characters (`_COMPACT_CONTENT_THRESHOLD`)
- Preserves the first 200 characters as a preview (`_COMPACT_PREVIEW_CHARS`)
- Returns the compacted message list via `CommandResult.metadata["compacted_messages"]`

**Factory:** `build_compact_command(pressure_monitor) -> CommandHandler`

## /status

```
/status
```

Comprehensive dashboard combining context window info, message counts, and memory summaries. Combines information from the pressure monitor and memory manager.

**Factory:** `build_status_command(pressure_monitor, memory_mgr) -> CommandHandler`

## /tools

```
/tools
```

Lists all registered tools with their risk levels, grouped by risk category (read-only, mutating, destructive).

**Factory:** `build_tools_command(tool_registry) -> CommandHandler`

## /skills

```
/skills
```

Lists all loaded skills with their names, descriptions, and tags.

**Factory:** `build_skills_command(skill_registry) -> CommandHandler`

## Registration

All built-in commands are registered by `build_command_dispatcher()` in `core/graph_builder/commands.py`:

```python
from langgraph_kit.core.graph_builder import build_command_dispatcher

dispatcher = build_command_dispatcher(
    memory_mgr=memory_mgr,
    pressure_monitor=pressure_monitor,
    tool_registry=tool_registry,  # optional
)
# Registers: /help, /memory, /context, /compact, /status, /tools, /skills
```
