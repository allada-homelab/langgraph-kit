# Faithfulness Evaluation

Given the user's input and the agent's output, evaluate whether the agent's
response is factually accurate and faithful to the information available.

## Input
{{input}}

## Output
{{output}}

## Scoring
- 1.0: Completely faithful, no hallucinations or fabricated information
- 0.7-0.9: Mostly faithful with minor inaccuracies that don't affect the core answer
- 0.4-0.6: Partially faithful, some unsupported claims or missing important caveats
- 0.1-0.3: Significant hallucinations or fabricated details
- 0.0: Entirely fabricated or contradicts the available information

Respond with a JSON object: {"score": <float 0-1>, "reason": "<brief explanation>"}
