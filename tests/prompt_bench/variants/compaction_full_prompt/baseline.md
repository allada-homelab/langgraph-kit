---
target: compaction.full_prompt
source: src/langgraph_kit/core/context_management/compaction.py:_FULL_COMPACTION_PROMPT
notes: |
  Baseline copy of the FULL compaction prompt. The ``{preamble}``
  placeholder is interpolated by the runtime with the no-tools
  preamble.
---
{preamble}

You are performing FULL conversation compaction. Summarize the ENTIRE conversation history.

First, produce a private <analysis> block to organize your thoughts about the important context.
Then, produce a <summary> block with the following JSON structure:

<summary>
{{
  "user_intent": "what the user originally asked for",
  "key_decisions": ["list of important decisions made"],
  "important_files": ["list of files that were read, modified, or are relevant"],
  "errors_and_fixes": ["list of errors encountered and how they were resolved"],
  "current_state": "what state the work is in right now",
  "pending_work": ["list of things still to be done"],
  "next_step": "the immediate next action that should be taken"
}}
</summary>

Preserve continuity-critical details: exact file paths, exact error messages, exact decisions.
Do not generalize or lose specifics that would be needed to resume the work.
