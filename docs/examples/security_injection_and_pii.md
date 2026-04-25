# Security — prompt injection + PII redaction

Inbound prompt-injection scanner (`scan_for_injection`) flags the
kit's catalogue of jailbreak / role-override / system-prompt-reveal
patterns. Outbound `redact()` rewrites PII / secret matches before the
message reaches the user.

```bash
uv run python -m examples.security_injection_and_pii
```

```python
--8<-- "examples/security_injection_and_pii.py"
```
