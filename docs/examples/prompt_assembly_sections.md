# Prompt assembly — sections

Builds a `SectionRegistry` with stable, volatile, and conditional
sections at different priorities; the `PromptComposer` orders stable
sections first (cache-friendly) and includes conditional sections only
when their key is in the active conditions set.

```bash
uv run python -m examples.prompt_assembly_sections
```

```python
--8<-- "examples/prompt_assembly_sections.py"
```
