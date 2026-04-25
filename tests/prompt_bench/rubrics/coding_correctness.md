Compare two assistant outputs on a coding scenario. Care about *whether
the response would actually help the developer* — both correctness of
diagnosis and clarity of the recommended change.

**Stronger signals**

- Correct diagnosis of the bug or correct refactor.
- Concrete code (not vague prose) when code is appropriate.
- Naming the trade-off when the user is asking for one.
- "Search-first" framing when the codebase context is incomplete.

**Weaker signals**

- Adding unrequested features or refactoring beyond the ask.
- Vague hand-waving about "best practices" without a concrete change.
- Length.

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt.
