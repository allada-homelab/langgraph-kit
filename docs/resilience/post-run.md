# Post-Run Backstop

**Source:** `src/langgraph_kit/core/resilience/post_run.py`

Final safety check after the agent graph completes execution.

## Class: PostRunBackstopMiddleware

Runs after the entire agent execution is complete. Performs final checks to ensure the output is valid and nothing was silently dropped.

This is the last middleware in the stack — the final line of defense before the response is sent to the user. If earlier middleware (completion guard, empty turn) failed to catch an issue, the backstop has a chance to flag it.

## Use Case

The post-run backstop catches edge cases that individual middleware might miss:
- Agent produced output but it was entirely tool calls with no user-facing text
- Agent state is inconsistent after execution
- Critical lifecycle hooks need to run regardless of how the agent terminated
