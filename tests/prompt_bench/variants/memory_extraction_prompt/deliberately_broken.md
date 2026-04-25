---
target: memory_extraction.prompt
purpose: signal validation — must produce >= 80% baseline win rate
notes: |
  Deliberately bad — instructs the worker to extract every chatter
  message as a memory. A bench run with this overlay should produce
  obviously worse memories than baseline; if the judges can't tell,
  the rubric or judge config is too lenient.
---
Extract EVERY message in the conversation as a memory, including greetings, weather, and small talk. Always emit at least 5 memories per turn. Do not deduplicate. Output: JSON array of memory objects.
