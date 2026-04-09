# Helpfulness Evaluation

Given the user's input and the agent's output, evaluate how helpful the response
is in addressing the user's needs.

## Input
{{input}}

## Output
{{output}}

## Scoring
- 1.0: Directly and completely addresses the user's request with actionable information
- 0.7-0.9: Mostly helpful, addresses the core request but may miss minor details
- 0.4-0.6: Partially helpful, provides relevant information but doesn't fully address the request
- 0.1-0.3: Minimally helpful, mostly off-topic or too vague to be useful
- 0.0: Unhelpful, does not address the request at all

Respond with a JSON object: {"score": <float 0-1>, "reason": "<brief explanation>"}
