# Plugins + skills

Loads a sample plugin from disk via `PluginLoader`, then discovers a
sample SKILL.md via `SkillRegistry`. Both subsystems sit at the kit's
"progressive disclosure" boundary: capability is loaded on demand
rather than baked into the agent's base prompt.

```bash
uv run python -m examples.plugins_skill_discovery
```

```python
--8<-- "examples/plugins_skill_discovery.py"
```

## Companion fixture

The plugin shape the loader expects — a `.py` file with a top-level
`contribute(**kwargs) -> PluginContribution` — lives in
[`examples/_sample_plugin.py`](https://github.com/allada-homelab/langgraph-kit/blob/main/examples/_sample_plugin.py).
The leading underscore excludes it from the smoke runner.
