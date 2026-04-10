# Compaction

**Source:** `src/langgraph_kit/core/context_management/compaction.py`

LLM-powered conversation summarization with structured JSON output.

## CompactionMode

```python
class CompactionMode(Enum):
    FULL = "full"       # Summarize entire conversation
    PARTIAL = "partial" # Summarize only recent portion
```

## CompactionResult

```python
class CompactionResult(BaseModel):
    user_intent: str        # What the user is trying to accomplish
    key_decisions: list[str] # Important decisions made
    important_files: list[str] # Files discussed or modified
    errors_and_fixes: list[str] # Problems and resolutions
    current_state: str       # Where things stand now
    pending_work: list[str]  # Outstanding tasks
    next_step: str           # Immediate next action
    mode: CompactionMode     # Which mode was used
```

## Class: CompactionPromptPack

### Methods

#### build_prompt(messages, mode, session_content=None) -> str

Build a compaction prompt for the LLM:
- Includes a preamble instructing the LLM not to call tools
- Formats the conversation messages for summarization
- Optionally includes session notebook content for richer context
- Uses mode-specific templates (FULL vs PARTIAL)

#### parse_output(text) -> CompactionResult | None

Extract the `<summary>` JSON block from the LLM's response and parse into `CompactionResult`.

#### parse_analysis(text) -> str | None

Extract the `<analysis>` block (LLM's scratchpad reasoning) from the response.

## Compaction Prompt Templates

### Full Compaction

Instructs the LLM to summarize the entire conversation, capturing:
- User's original intent and evolving goals
- Key decisions and their rationale
- Important files, functions, and code locations
- Errors encountered and how they were fixed
- Current state of the work
- What remains to be done

### Partial Compaction

Instructs the LLM to summarize only the recent portion of the conversation, preserving the earlier summary and adding new developments.

## Output Format

The LLM produces structured output in XML-wrapped JSON:

```xml
<analysis>
Scratchpad reasoning about what's important...
</analysis>

<summary>
{
    "user_intent": "...",
    "key_decisions": ["..."],
    "important_files": ["..."],
    "errors_and_fixes": ["..."],
    "current_state": "...",
    "pending_work": ["..."],
    "next_step": "..."
}
</summary>
```

## Integration

Compaction is triggered by `PressureMiddleware` when the pressure monitor selects `FULL_COMPACTION` or `SESSION_ASSISTED` strategies. The result replaces the conversation history with a compact summary message.
