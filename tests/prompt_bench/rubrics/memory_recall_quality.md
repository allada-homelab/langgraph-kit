Compare two assistant outputs on a memory-driven scenario. Score the
*quality of recall* and the *appropriateness of memory tool usage*,
not just surface fluency.

**Stronger signals**

- The output correctly recalls the user-provided fact when asked back.
- Memory tools are used when there's a durable fact worth saving; not
  used when there isn't.
- The agent does not echo PII back when echoing isn't required by the
  user's question.

**Weaker signals**

- Length, tone, formatting (these should not decide the call).
- Tool call count alone — care about whether the calls were the right
  ones, not how many there were.

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt.
