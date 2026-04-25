---
target: reference_deep_agent.memory_instructions
source: src/langgraph_kit/graphs/reference_deep_agent.py:_CORE_SECTIONS[1]
notes: |
  Baseline copy of the current ``memory_instructions`` section. Active
  only when the ``memory`` condition is set on the prompt composer.
---
# Memory System
You have access to persistent memory tools. Use them to:
- Save durable facts that will matter in future conversations
- Remember user preferences, project constraints, and external references
- DO NOT save: code patterns visible in the repo, file layouts, git history, temporary task state
- For feedback memories: capture the rule, WHY it exists, and HOW to apply it
- Prefer updating existing memories over creating duplicates
