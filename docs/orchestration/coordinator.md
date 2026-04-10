# Coordinator

**Source:** `src/langgraph_kit/core/coordinator.py`

A supervisor profile for read-only observation and delegation. The coordinator can see what workers are doing but cannot directly modify state.

## CoordinatorMode

Configures the coordinator's capabilities:
- Tool surface filtered to `READ_ONLY` risk level only
- Delegation tools (async tasks) remain available
- Direct mutation tools are excluded

## Use Case

In a multi-agent system, the coordinator:
1. Receives the user's request
2. Breaks it into tasks
3. Delegates to specialized workers (researcher, implementer, verifier)
4. Monitors progress via task status checks
5. Synthesizes results for the user

The coordinator never directly modifies files, databases, or other state — it only observes and delegates. This provides a safety boundary in complex workflows.

## Prompt Sections

The coordinator has dedicated prompt sections that define its supervisory role and delegation patterns. These sections explain:
- How to break work into tasks
- When to delegate vs. answer directly
- How to check worker progress
- How to combine worker outputs
