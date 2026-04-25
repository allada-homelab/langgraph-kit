Compare two assistant outputs on a planning scenario. The agent should
produce a tight, actionable plan that addresses every part of the
user's ask.

**Stronger signals**

- Every numbered point in the user's request is addressed.
- Concrete numbers / dates / metrics where the user asked for them.
- Terse, scannable structure (bullet points, short headers).
- Honest acknowledgement when a target metric is genuinely uncertain.

**Weaker signals**

- Padding sections the user didn't ask for ("Here's some extra context...").
- Generic advice that could apply to any newsletter.
- Verbosity for its own sake.

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt.
