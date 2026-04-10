# Prompt Assembly Overview

The prompt assembly system composes the agent's system prompt from modular sections and dynamic context providers. Sections are classified by stability to maximize Anthropic prompt cache hits.

## Components

| Module | Purpose |
|--------|---------|
| [Sections](sections.md) | `PromptSection`, `SectionStability`, `SectionRegistry` |
| [Composer](composer.md) | `PromptComposer` assembly pipeline |
| [Context Providers](context-providers.md) | Dynamic runtime context injection |

## How It Works

```
SectionRegistry (static sections)
    │
    ├── STABLE sections (rarely change)
    │   └── core_identity, memory_instructions, ...
    │
    ├── VOLATILE sections (change every turn)
    │   └── thread context, memory summaries, ...
    │
    └── CONDITIONAL sections (active based on predicate)
        └── coding_workflow (only when coding profile active)
    │
    ▼
PromptComposer.compose(context)
    │
    ├── 1. Collect active sections
    ├── 2. Sort: STABLE first, then by priority
    ├── 3. Check PromptCache for stable section block
    ├── 4. Append volatile sections
    ├── 5. Run ContextProviders for dynamic content
    └── 6. Join into final system prompt
    │
    ▼
Final system prompt string
```

## Why Stable-First Ordering?

Anthropic's prompt caching works by caching a prefix of the prompt. If the prefix is identical between requests, the cached version is used. By placing stable (rarely-changing) sections first, the cache hit rate improves dramatically:

```
[STABLE: core identity]         ← cached prefix
[STABLE: memory instructions]   ← cached prefix
[STABLE: tool guidance]         ← cached prefix
---
[VOLATILE: thread context]      ← changes each turn
[VOLATILE: memory summaries]    ← changes as memories update
[CONDITIONAL: coding workflow]  ← may or may not be present
```

The stable prefix stays in the cache across turns, saving tokens and reducing latency.
