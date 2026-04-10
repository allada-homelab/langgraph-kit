# CLI Reference

**Source:** `src/langgraph_kit/cli.py`

langgraph-kit includes a CLI for scaffolding new agents from templates.

## Usage

```bash
uv run python -m langgraph_kit.cli <command> [args]
```

## Commands

### new

Generate a new agent file from a template.

```bash
uv run python -m langgraph_kit.cli new <agent_id> [--output-dir <path>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `agent_id` | Yes | — | ID for the new agent (e.g., `my-agent`) |
| `--output-dir` | No | `.` | Directory to write the generated file |

**Output:** Creates `<agent_id>.py` with a full agent template including:
- Prompt section definitions (core identity, etc.)
- Worker definitions (researcher, implementer, verifier)
- Tool registration using builder utilities
- Middleware stack construction
- Backend factory setup
- Command dispatcher configuration
- `build_graph()` function following the standard contract

**Example:**

```bash
uv run python -m langgraph_kit.cli new code-reviewer --output-dir backend/src/app/agents/graphs/
```

### list

Show available agent templates.

```bash
uv run python -m langgraph_kit.cli list
```

Currently shows one template: `default`.

## Generated Agent Structure

The generated agent file includes commented sections you can customize:

```python
# Prompt sections — customize the agent's identity and instructions
_CORE_SECTIONS = [
    PromptSection(id="core_identity", content="...", stability=SectionStability.STABLE),
    ...
]

# Worker definitions — customize sub-agent roles
WORKER_DEFINITIONS = [
    {"name": "researcher", "system_prompt": "..."},
    {"name": "implementer", "system_prompt": "..."},
    {"name": "verifier", "system_prompt": "..."},
]

# Build function — the entry point
def build_graph(checkpointer, store):
    ...
```

After generating, register the agent in `graphs/__init__.py` to include it in `register_all()`.
