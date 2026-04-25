# Disaster recovery — export & import

JSON Lines export of selected store namespaces, round-tripped through
`DisasterRecoveryManager.import_jsonl` with `ImportMode.REPLACE`.
Complement to full database backups; useful for selective restore.

```bash
uv run python -m examples.disaster_recovery_export_import
```

```python
--8<-- "examples/disaster_recovery_export_import.py"
```
