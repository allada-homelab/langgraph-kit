# FastAPI router — full endpoint reference (curl recipes)

The demo at [`examples/fastapi_minimal_router.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/fastapi_minimal_router.py)
mounts the router and exercises two of the eleven endpoints in-process.
This document complements it: each `curl` snippet below targets one of
the remaining endpoints exposed by `create_agent_router(...)`. Run the
minimal demo first to confirm the wiring works, then swap to a live
server (e.g. `uvicorn examples.fastapi_minimal_router:app --port 8000`)
to exercise these recipes end-to-end.

Prefix every endpoint with the same `prefix=` you passed to
`app.include_router(...)`. The recipes below assume `/api/v1`.

## Discovery

```bash
curl -sS http://localhost:8000/api/v1/agents/
```

## Streaming (SSE)

```bash
curl -N -sS -X POST http://localhost:8000/api/v1/agents/echo/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

The stream yields `id:` + `data:` chunks ending with `data: [DONE]`.
See [`examples/streaming_sse_events.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/streaming_sse_events.py)
for a parser.

## Synchronous invoke

```bash
curl -sS -X POST http://localhost:8000/api/v1/agents/echo/invoke \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Returns `{"thread_id": "...", "response": "..."}`.

## Thread state + branching

```bash
# Inspect current state
curl -sS http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/state

# List checkpoints
curl -sS http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/checkpoints

# Branch from a specific checkpoint
curl -sS -X POST http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/branch \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint_id": "<ckpt-id>"}'
```

## HITL resume

```bash
# When the agent is paused on an interrupt, the SSE stream emits an
# {"interrupt": {...}} chunk. Resume with:
curl -sS -X POST http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/resume \
  -H 'Content-Type: application/json' \
  -d '{"responses": [{"type": "accept"}]}'
```

See [`examples/hitl_approval_flow.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/hitl_approval_flow.py)
for the in-process equivalent (no real server).

## Message queue

```bash
# Enqueue a message for a busy thread
curl -sS -X POST http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/queue \
  -H 'Content-Type: application/json' \
  -d '{"message": {"role": "user", "content": "follow-up question"}}'

# List queued
curl -sS http://localhost:8000/api/v1/agents/echo/threads/<thread-id>/queue
```

## Authentication

The router requires a `get_current_user` FastAPI dependency. The
in-process demo provides a fixed-user stub. Production deployments
should plug in an OAuth / API-key dependency that resolves the current
user from request headers; the resolved object must have ``id`` and
``email`` attributes (the
[`UserInfo`](https://github.com/allada-homelab/langgraph-kit/blob/main/src/langgraph_kit/observability.py)
protocol).
