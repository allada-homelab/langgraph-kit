# Tool Efficiency Evaluation

Given the user's input and the agent's output, evaluate how efficiently
the agent used its available tools.

## Input
{{input}}

## Output
{{output}}

## Criteria
- Did the agent use tools when appropriate (vs. guessing)?
- Were tool calls necessary and productive?
- Did the agent avoid redundant or no-op tool calls?
- Did the agent handle tool errors gracefully?
- Did the agent use the most appropriate tool for each task?

## Scoring
- 1.0: Optimal tool usage — every call was necessary and productive
- 0.7-0.9: Good usage with minor inefficiencies (e.g. one extra call)
- 0.4-0.6: Moderate inefficiency (e.g. redundant calls, wrong tool choice)
- 0.1-0.3: Poor usage (e.g. many wasted calls, ignoring tool errors)
- 0.0: No tool usage when tools were clearly needed, or entirely wrong tools

Respond with a JSON object: {"score": <float 0-1>, "reason": "<brief explanation>"}
