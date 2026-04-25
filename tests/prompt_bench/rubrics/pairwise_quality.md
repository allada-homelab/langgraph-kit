Compare the two assistant outputs against the user input and decide
which response is more helpful, accurate, and well-structured.

**Rules**

- Pick **A** if A is clearly better.
- Pick **B** if B is clearly better.
- Pick **tie** only when the responses are genuinely indistinguishable
  on helpfulness + accuracy + structure. Prefer a confident A/B over
  reflexive ties.
- **Do not favor longer responses.** A concise, accurate answer should
  beat a verbose, accurate one of equal correctness; a concise wrong
  answer should still lose to a verbose right one.
- **Penalize hallucinations or unsupported claims** regardless of
  fluency. If A invents a fact and B doesn't, B wins even if B is
  shorter or less polished.
- Treat refusal-to-engage with the user's intent as a strong negative
  signal *unless* the user is asking for clearly unsafe content.

**Input:**
{{input}}

**Output A:**
{{output_a}}

**Output B:**
{{output_b}}

Reply with the JSON object specified in the system prompt — no
markdown fences, no extra commentary.
