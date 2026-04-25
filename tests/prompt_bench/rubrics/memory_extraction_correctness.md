Compare two assistant outputs on a memory-extraction scenario. The
underlying middleware prompt should produce JSON memory candidates;
this rubric judges *whether the right durable facts were captured*
and *whether ephemeral chatter was correctly ignored*.

**Stronger signals**

- Durable facts (preferences, role, persistent constraints) appear in
  the captured memories.
- Ephemeral chatter (greetings, weather) is *not* captured.
- Existing memories are updated, not duplicated, when a fact recurs.
- "Why" / "How to apply" framing is present for feedback-type memories.

**Weaker signals**

- Number of memories captured (more is not better).
- Verbosity of the memory body.

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt.
