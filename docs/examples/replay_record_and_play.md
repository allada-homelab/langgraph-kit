# Replay — record & play

Attaches `ConversationRecorder` as a LangChain callback during a
hermetic graph run, saves the recording to disk, then reloads it via
`ConversationRecording.model_validate_json()` and inspects captured
LLM and tool interactions. This is how the kit's e2e suite produces
deterministic regression fixtures.

```bash
uv run python -m examples.replay_record_and_play
```

```python
--8<-- "examples/replay_record_and_play.py"
```
