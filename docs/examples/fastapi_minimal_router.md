# FastAPI — minimal router

Mounts `create_agent_router(...)` onto a FastAPI app and drives it via
`httpx.AsyncClient` with `ASGITransport`, so the demo runs hermetically
without binding a port. Hits the discovery endpoint
(`GET /agents/`) and the `invoke` endpoint to keep output predictable.
The full router exposes 11 endpoints (stream / state / resume / branch /
queue / etc.) — see `tests/e2e/test_fastapi_e2e.py` for the wider
surface.

```bash
uv run python -m examples.fastapi_minimal_router
```

```python
--8<-- "examples/fastapi_minimal_router.py"
```
