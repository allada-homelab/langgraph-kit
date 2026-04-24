# Graceful Shutdown

Langgraph-kit drains in-flight background work during process shutdown so long-running agent runs finish cleanly — no half-written `Store` records, no lost Langfuse spans, no stuck `async_task` rows marked `RUNNING` forever.

## What gets drained

The lifespan teardown in [`contrib.fastapi.create_app_lifespan`](../integrations/fastapi.md) calls `drain_background_tasks(timeout)` before closing MCP connections, flushing Langfuse, and releasing the persistence context.

The drain target is every [async sub-agent task](../orchestration/async-tasks.md) started via `AsyncTaskManager.start()` — these outlive the HTTP request that kicked them off and are the only work the HTTP layer cannot drain at the socket level.

Draining does **not** cover:

- In-flight HTTP requests — Uvicorn/Gunicorn already hold the connection open until the handler returns, so lifespan shutdown only runs once every active handler has finished.
- Thread-level busy locks (`ThreadBusyTracker`) — these are advisory and auto-expire after 10 minutes; they need no explicit drain.

## Timeout behavior

A single `AgentConfig` field controls the wait:

```python
from langgraph_kit import AgentConfig, configure

configure(AgentConfig(
    shutdown_timeout_seconds=30.0,  # default
))
```

- `> 0` — wait up to this many seconds for tasks to finish on their own, then cancel the rest.
- `0` — cancel immediately (a ~10 ms window is still given so cancelled tasks can update their Store record to `CANCELLED`).
- Negative values skip draining entirely (the lifespan teardown proceeds to MCP close and Langfuse flush without waiting).

A task that is cancelled by the drain has its Store record updated from `RUNNING` to `CANCELLED`; readers checking task status after shutdown see the real terminal state instead of a stuck row.

## Production checklist

For Kubernetes, systemd, or any orchestrator that sends SIGTERM on shutdown:

1. **Match the pod / service terminationGracePeriod to `shutdown_timeout_seconds` + headroom.** If your tasks typically take 20 s and `shutdown_timeout_seconds = 30`, set `terminationGracePeriodSeconds = 45` so the process has time to drain *and* flush Langfuse after.
2. **Configure the load balancer / service mesh to stop routing first.** SIGTERM triggers lifespan shutdown, which includes the drain; any request that lands during the drain will wait on the server socket. Routing cutoff at the edge avoids that.
3. **Tune `shutdown_timeout_seconds` to the 95th percentile of your sub-agent runs.** Setting it too low cancels useful work; setting it unreasonably high lets a single hung task hold up the whole pod's shutdown.
4. **Check logs after rollouts.** The lifespan emits one line summarising the drain:
   ```
   shutdown drain: 4 async task(s) completed, 1 cancelled
   ```
   If `cancelled` is consistently > 0 in steady state, the timeout is too short for your workload.

## Signal handling

Langgraph-kit does not install its own signal handlers. It relies on the ASGI server (Uvicorn, Hypercorn, Gunicorn) to translate SIGTERM/SIGINT into a lifespan shutdown event. That event runs the cleanup block in `create_app_lifespan`, which is where the drain happens.

If you embed the app in a custom entrypoint, make sure the shutdown pathway exits the `lifespan` context manager — otherwise the drain never runs.
