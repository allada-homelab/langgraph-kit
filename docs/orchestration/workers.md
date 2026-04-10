# Worker Definitions

**Source:** `src/langgraph_kit/core/orchestration/workers.py`

Declarative worker (sub-agent) definitions compatible with deepagents' `subagents` parameter. Agent graph builders compose their worker list from these shared definitions.

## General-Purpose Workers (R0)

### RESEARCHER_DEFINITION

```python
{
    "name": "researcher",
    "description": "Deep codebase research and investigation...",
    "system_prompt": "You are a research specialist. Investigate thoroughly..."
}
```

Role: Explore files, trace execution paths, understand architecture. **Does not make changes** — only reports findings with file paths and line numbers.

### IMPLEMENTER_DEFINITION

```python
{
    "name": "implementer",
    "description": "Focused code implementation within a bounded scope...",
    "system_prompt": "You are an implementation specialist. Make the requested changes..."
}
```

Role: Execute well-understood changes precisely and completely. Follows existing code conventions.

### VERIFIER_DEFINITION

```python
{
    "name": "verifier",
    "description": "Independent verification of changes...",
    "system_prompt": "You are a verification specialist. Review the changes..."
}
```

Role: Check correctness, edge cases, and adherence to requirements. **Does not fix issues** — reports them for the supervisor.

## Coding-Profile Workers (R2)

### CODING_VERIFIER_DEFINITION

An enhanced verifier for coding agents with structured output format:

```python
{
    "name": "verifier",
    "description": "Independent verification of code changes...",
    "system_prompt": "You are a SKEPTICAL verification specialist..."
}
```

Key differences from the standard verifier:
- **Skeptical approach** — assumes nothing works until evidence shows it does
- **Reads actual files** — does not rely on summaries or claims
- **Runs tests** if available
- **Structured output**: `PASS`, `WARN`, `FAIL` findings with file paths and line numbers

## Pre-Composed Worker Lists

```python
from langgraph_kit.core.orchestration import R0_WORKERS, CODING_WORKERS
```

### R0_WORKERS

```python
[RESEARCHER_DEFINITION, IMPLEMENTER_DEFINITION, VERIFIER_DEFINITION]
```

### CODING_WORKERS

```python
[RESEARCHER_DEFINITION, IMPLEMENTER_DEFINITION, CODING_VERIFIER_DEFINITION]
```

## Verification Re-Export

**Source:** `src/langgraph_kit/core/orchestration/verification.py`

For convenience, `CODING_VERIFIER_DEFINITION` is re-exported from `verification.py`:

```python
from langgraph_kit.core.orchestration.verification import CODING_VERIFIER_DEFINITION
```

## Usage in Agent Builders

```python
from langgraph_kit.core.orchestration import R0_WORKERS, CODING_WORKERS

# In R0 agent builder
graph = create_deep_agent(llm, subagents=R0_WORKERS, ...)

# In coding agent builder
graph = create_deep_agent(llm, subagents=CODING_WORKERS, ...)
```
