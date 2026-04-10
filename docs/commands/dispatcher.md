# Command Dispatcher

**Source:** `src/langgraph_kit/core/commands/dispatch.py`

## Types

### CommandResult

```python
@dataclass
class CommandResult:
    output: str           # Text output to show the user
    handled: bool         # Whether the command was handled
    metadata: dict = {}   # Optional metadata (e.g., compacted_messages)
```

### CommandInfo

```python
@dataclass
class CommandInfo:
    name: str             # Command name (e.g., "/help")
    description: str      # Human-readable description
    usage: str            # Usage pattern (e.g., "/memory [scope]")
```

### CommandHandler

```python
CommandHandler = Callable[[str, dict], Awaitable[CommandResult]]
```

Async function taking `(args_string, context_dict)` and returning `CommandResult`.

## Class: CommandDispatcher

### Methods

#### register(name, handler, *, description="", usage="")

Register a command handler with metadata.

```python
dispatcher.register(
    "/greet",
    greet_handler,
    description="Greet the user",
    usage="/greet [name]",
)
```

#### is_command(text) -> bool

Check if text starts with a registered command name.

#### dispatch(text, context=None) -> CommandResult

Parse the command and args from text, execute the matching handler.

```python
result = await dispatcher.dispatch("/memory user", {"store": store})
if result.handled:
    print(result.output)
```

#### list_commands() -> list[CommandInfo]

Return metadata for all registered commands.

## Custom Commands

```python
async def my_handler(args: str, context: dict) -> CommandResult:
    return CommandResult(output=f"You said: {args}", handled=True)

dispatcher.register("/echo", my_handler, description="Echo input")
```
