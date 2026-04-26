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

### openapi

Dump the FastAPI agent router's OpenAPI specification to stdout or a file. Useful for generating typed clients without spinning up a live server.

```bash
uv run python -m langgraph_kit.cli openapi [--output spec.json] [--indent 2]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--output` | No | stdout | Write the spec to this path instead of stdout |
| `--indent` | No | `2` | JSON indent for the dumped spec (`0` for compact) |

**What gets exported:** the full router from `langgraph_kit.contrib.fastapi.create_agent_router()` mounted on a temporary `FastAPI` app. The spec covers `/agents`, `/invoke`, `/stream`, and any other routes the router declares, with all Pydantic request/response models referenced under `components.schemas`.

#### Generating a typed client

The dumped spec feeds into any OpenAPI client generator. Two recipes — pick the one that matches your stack.

**Python client via [`openapi-python-client`](https://github.com/openapi-generators/openapi-python-client):**

```bash
# 1. Dump the spec.
uv run python -m langgraph_kit.cli openapi --output spec.json

# 2. Generate a typed client package next to your app.
uvx --from openapi-python-client openapi-python-client generate --path spec.json
```

The generator writes a package whose `Client` class wraps every endpoint with typed `attrs`-style models. Drop it into your app and call `Client(base_url=...).agents_get_invoke(...)`.

**TypeScript / JS client via [`openapi-generator-cli`](https://github.com/OpenAPITools/openapi-generator-cli):**

```bash
uv run python -m langgraph_kit.cli openapi --output spec.json
npx @openapitools/openapi-generator-cli generate \
    -i spec.json \
    -g typescript-fetch \
    -o ./gen/langgraph-kit-client
```

#### Notes

- **SSE doesn't model cleanly in OpenAPI.** The `/stream` endpoint declares `text/event-stream` as its response content type; concrete SSE event payload schemas are referenced from `components.schemas` so generated clients can still type the events even though the transport itself is opaque.
- **No live server required.** The CLI mounts the router on a temporary `FastAPI()` and calls `app.openapi()`; nothing binds to a port. Safe to run in CI to keep a generated SDK up to date.

### shell

Interactive REPL for a registered agent. See `langgraph-kit shell --help` for the full flag list.

| Slash command | Effect |
|---------------|--------|
| `/exit`, `/quit`, `/q` | End the session (or `Ctrl-D` / `Ctrl-C`). |
| `/info` | Print the active agent id, thread id, user id, and module path without invoking the agent. |
