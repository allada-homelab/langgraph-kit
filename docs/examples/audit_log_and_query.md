# Audit log

Append-only `(actor, action, target, metadata)` rows via `AuditStore`,
queryable by actor / action / time window. Newest-first, monthly
buckets so listing stays cheap.

```bash
uv run python -m examples.audit_log_and_query
```

```python
--8<-- "examples/audit_log_and_query.py"
```
