# Rate limit — token bucket

Per-key token bucket with continuous refill via
`InMemoryRateLimitBackend`. Same backend powers `RateLimitMiddleware`
in front of the FastAPI router. Multi-worker deployments need a
cross-process backend (Redis, etc.).

```bash
uv run python -m examples.rate_limit_token_bucket
```

```python
--8<-- "examples/rate_limit_token_bucket.py"
```
