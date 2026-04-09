# Task Completion Evaluation

Given the user's input and the agent's output, evaluate whether the agent
successfully completed the requested task.

## Input
{{input}}

## Output
{{output}}

## Scoring
- 1.0: Task fully completed with all requirements met
- 0.7-0.9: Task mostly completed, minor requirements may be missing
- 0.4-0.6: Task partially completed, significant parts remain undone
- 0.1-0.3: Task barely started or mostly failed
- 0.0: Task not attempted or completely wrong approach

Consider whether the agent:
1. Understood the task correctly
2. Used appropriate tools and methods
3. Produced a complete result
4. Handled edge cases or errors gracefully

Respond with a JSON object: {"score": <float 0-1>, "reason": "<brief explanation>"}
