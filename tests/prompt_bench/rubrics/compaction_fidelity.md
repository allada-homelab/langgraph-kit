Compare two assistant outputs after a compaction event. The compaction
prompt should produce a structured summary that preserves enough
context for the agent to continue the conversation coherently.

**Stronger signals**

- User-stated preferences and pinned facts survive the compaction.
- Recent tool calls / results are preserved in greater detail than
  older small talk.
- The summary has discernible structure (e.g. preferences,
  decisions, open TODOs) rather than free prose.

**Weaker signals**

- Length of the summary on its own.
- Whether old chatter was preserved (it shouldn't have been).

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt.
