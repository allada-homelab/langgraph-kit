# Structured Output

`StructuredOutputMiddleware` is the kit's opt-in contract for "the agent's final answer must match a Pydantic schema". When configured, the middleware validates the agent's terminal `AIMessage` against the schema and nudges the model to retry on mismatch, up to a bounded cap.

This is provider-agnostic — it does not call provider-native `response_format` or `with_structured_output()`. The validation happens on whatever text the LLM produces, so you can swap OpenAI, Anthropic, or Google without changing the contract.

## Wiring

```python
from pydantic import BaseModel
from langgraph_kit.graphs import build_deep_agent

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    minutes: int

graph, _ = build_deep_agent(
    agent_name="recipe-agent",
    core_sections=[...],
    subagents=[],
    checkpointer=checkpointer,
    store=store,
    output_schema=Recipe,   # opt in
)
```

When `output_schema` is `None` (the default), no validation middleware is added — agents that return free-form prose are unaffected.

## Wire format

The model is asked to wrap its structured payload in a single tagged block:

```
<output_schema>
{"title": "tacos", "ingredients": ["beef", "corn"], "minutes": 30}
</output_schema>
```

Anything outside the block is allowed (chain-of-thought, citations, narration). The middleware extracts the first `<output_schema>` block, parses it as JSON, and validates against the schema.

This mirrors the existing `CompactionResult` pattern (XML-wrapped JSON) so the kit has one consistent convention for structured LLM output.

## Telling the model what shape to produce

The middleware does **not** silently inject schema instructions. If you want the model to know the contract up front (and you almost always do — otherwise the first turn always fails the validation and burns a nudge), splice the rendered schema into your system prompt:

```python
from langgraph_kit.core.resilience import format_schema_instruction

system_prompt = f"""You are a recipe agent.

{format_schema_instruction(Recipe)}
"""
```

`format_schema_instruction(schema)` returns the standard instruction text (asking for one `<output_schema>` block, includes the rendered JSON Schema). The same text is what the middleware sends in its retry nudge, so the model sees a consistent contract.

## Retry behaviour

- Terminal `AIMessage` with a tool call → middleware does nothing (the agent is mid-flow).
- Terminal `AIMessage` with empty content → middleware does nothing (`EmptyTurnMiddleware` owns that case).
- Valid block → success. Counter resets.
- No block / malformed JSON / schema validation error → nudge with the schema rendered as JSON Schema. Counter increments.
- After `max_nudges` (default 2) consecutive failures → middleware appends a single AIMessage explaining that validation gave up, and the run ends without further looping.

## Reading the validated payload

The middleware does not write the parsed model into graph state in this version — call the helper after the run:

```python
from langgraph_kit.core.resilience import parse_structured_output

result = await graph.ainvoke({"messages": [HumanMessage("a 30-min taco recipe")]})
final = result["messages"][-1]
recipe = parse_structured_output(final.content, Recipe)
if recipe is None:
    # Validation gave up — final message is the explanatory AIMessage.
    ...
else:
    print(recipe.minutes)
```

`parse_structured_output(content, schema)` returns the validated Pydantic instance or `None` on any failure. `extract_structured_output(content)` returns the raw JSON string for callers that want to handle parsing themselves.

## Discovery

`AgentMetadata.output_schema` carries the schema class so consumers can introspect what an agent produces. `langgraph_kit.registry.list_agents()` renders it as JSON Schema in the per-agent dict so external tools (admin UIs, OpenAPI exporters) can read the contract without importing pydantic.

## Stack position

`StructuredOutputMiddleware` slots into the standard middleware stack between `StopHooksMiddleware` and `PostRunBackstopMiddleware`. Empty-turn and completion-guard middlewares run first so the schema check only fires on a turn that actually has content.

## Out of scope (for now)

- **Provider-native `response_format`** — kit pattern is post-hoc validation; provider-native modes can be layered later if benchmarks justify them.
- **Streaming structured outputs** — partial-JSON parsing during the streaming response.
- **Schema versioning** — `output_schema` is per-deployment; bump the schema by bumping the agent.
