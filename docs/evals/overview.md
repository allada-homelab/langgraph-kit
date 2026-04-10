# Evaluation Framework

**Source:** `src/langgraph_kit/evals/`

The evaluation framework provides tools for measuring agent quality through rule-based and model-graded metrics.

## Components

| Module | Purpose |
|--------|---------|
| `models.py` | Evaluation result data models |
| `runner.py` | Test harness for running evaluations |
| `report.py` | Report generation from evaluation results |
| `metrics/rule_based.py` | Deterministic metric calculators |
| `metrics/model_graded.py` | LLM-graded metrics using structured prompts |

## Evaluation Types

### Rule-Based Metrics

Deterministic checks that don't require an LLM:
- Response length within bounds
- Required keywords present
- Format compliance (JSON, markdown, etc.)
- Tool usage patterns (correct tools called)

### Model-Graded Metrics

LLM-powered quality assessment:
- Response relevance and accuracy
- Instruction following
- Code quality (when applicable)
- Tone and helpfulness

## Running Evaluations

```python
from langgraph_kit.evals.runner import EvalRunner

runner = EvalRunner(graph, config)
results = await runner.run(test_cases)
```

## Report Generation

```python
from langgraph_kit.evals.report import generate_report

report = generate_report(results)
print(report.summary)
```

## Use Case

The evaluation framework helps you:
- Validate agent quality before deployment
- Compare agent configurations
- Detect regressions when changing prompts or tools
- Benchmark custom agents against built-in ones
