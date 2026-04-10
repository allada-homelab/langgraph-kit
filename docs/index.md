# langgraph-kit Documentation

**langgraph-kit** is a reusable LangGraph agent toolkit for building production-grade AI agents with persistent memory, rich tool capabilities, multi-agent orchestration, and streaming support.

Licensed under AGPL-3.0-or-later. Requires Python 3.13.

---

## Contents

### Getting Started

- [Installation](getting-started/installation.md) — Installing the package and optional extras
- [Configuration](getting-started/configuration.md) — Setting up AgentConfig for your environment
- [Quickstart](getting-started/quickstart.md) — Building and running your first agent in minutes

### Architecture

- [Overview](architecture/overview.md) — High-level design, package layout, and data flow
- [Design Principles](architecture/design-principles.md) — Conventions, patterns, and trade-offs

### Core Modules

- [Configuration](core/config.md) — `AgentConfig`, `configure()`, and `get_config()`
- [LLM Factory](core/llm.md) — Multi-provider LLM instantiation (OpenAI, Anthropic, Google)
- [Persistence](core/persistence.md) — Checkpointer and Store factory (PostgreSQL / SQLite)
- [Registry](core/registry.md) — In-memory agent registration and lookup
- [Streaming](core/streaming.md) — SSE event streaming with `astream_events` v2
- [Observability](core/observability.md) — Langfuse tracing and run config building

### Memory System

- [Overview](memory/overview.md) — Memory taxonomy, scopes, and storage model
- [Memory Models](memory/models.md) — `MemoryRecord`, `MemoryType`, `MemoryScope`
- [Persistent Manager](memory/persistent-manager.md) — CRUD facade over LangGraph Store
- [Agent Memory](memory/agent-memory.md) — Worker-scoped memory for specialized agents
- [Auto-Extraction](memory/extraction.md) — LLM-powered post-turn memory extraction
- [Session Notebook](memory/session-notebook.md) — Thread-local structured notebook
- [Consolidation](memory/consolidation.md) — Background memory merging, pruning, and normalization
- [Shared Memory](memory/shared-memory.md) — Team scope publishing with secret detection

### Tools & Capabilities

- [Overview](tools/overview.md) — Tool capability model and risk taxonomy
- [Tool Capability](tools/capability.md) — `ToolCapability` dataclass and `ToolRisk` enum
- [Tool Registry](tools/registry.md) — Registration, filtering, compilation, and prompt fragments
- [Memory Tools](tools/memory-tools.md) — Agent-callable CRUD tools for persistent memory
- [Deferred Tools](tools/deferred-tools.md) — Lazy tool discovery and registration
- [Worktree Tools](tools/worktree-tools.md) — Git worktree isolation for coding agents

### Command System

- [Overview](commands/overview.md) — Slash-command dispatch architecture
- [Dispatcher](commands/dispatcher.md) — `CommandDispatcher` routing and execution
- [Middleware](commands/middleware.md) — `CommandMiddleware` for intercepting commands pre-LLM
- [Built-in Commands](commands/builtins.md) — `/help`, `/memory`, `/context`, `/compact`, `/status`, `/tools`, `/skills`

### Context Management

- [Overview](context-management/overview.md) — Context pressure, compaction, and continuation
- [Pressure Monitor](context-management/pressure.md) — Token estimation and mitigation strategies
- [Compaction](context-management/compaction.md) — Conversation summarization (full and partial)
- [Continuation](context-management/continuation.md) — Token-budget continuation with diminishing-returns detection

### Prompt Assembly

- [Overview](prompt-assembly/overview.md) — Layered prompt composition with caching
- [Sections](prompt-assembly/sections.md) — `PromptSection`, `SectionStability`, `SectionRegistry`
- [Composer](prompt-assembly/composer.md) — `PromptComposer` assembly pipeline
- [Context Providers](prompt-assembly/context-providers.md) — Dynamic runtime context injection

### Orchestration

- [Overview](orchestration/overview.md) — Multi-agent coordination patterns
- [Workers](orchestration/workers.md) — Declarative worker (sub-agent) definitions
- [Async Tasks](orchestration/async-tasks.md) — Fire-and-forget background sub-agents
- [Message Queue](orchestration/queue.md) — Store-backed per-thread message queue
- [Coordinator](orchestration/coordinator.md) — Supervisor profile with read-only delegation

### Resilience

- [Overview](resilience/overview.md) — Error recovery and completion guards
- [Completion Guard](resilience/completion-guard.md) — Heuristic premature-completion detection
- [Empty Turn](resilience/empty-turn.md) — Nudging the model when no output is produced
- [Tool Error](resilience/tool-error.md) — Structured error handling with transient retry
- [Post-Run Backstop](resilience/post-run.md) — Final checks after graph execution

### Human-in-the-Loop

- [Overview](hitl/overview.md) — Interrupt-based approval for destructive operations
- [Models](hitl/models.md) — `ActionRequest`, `HumanInterrupt`, `HumanResponse`, `ResumeRequest`
- [Tools](hitl/tools.md) — `approve_action` tool and interrupt mechanics

### UI Events & Artifacts

- [Overview](ui-events/overview.md) — Rich frontend events alongside streaming tokens
- [Artifacts](ui-events/artifacts.md) — Structured code, markdown, diagrams, and tables
- [Progress & Suggestions](ui-events/progress-suggestions.md) — Ephemeral UI events

### Plugins & MCP

- [Overview](plugins/overview.md) — Plugin system and MCP integration
- [MCP Adapter](plugins/mcp-adapter.md) — Wrapping MCP tools as native `ToolCapability`
- [Plugin Loader](plugins/plugin-loader.md) — Python plugin files with `contribute()` functions

### Skills

- [Overview](skills/overview.md) — Progressive skill disclosure from SKILL.md files
- [Skill Registry](skills/registry.md) — Discovery, indexing, and search
- [Defining Skills](skills/defining-skills.md) — Writing SKILL.md files with YAML frontmatter

### Agent Graphs

- [Overview](agents/overview.md) — Agent graph contract and registration
- [Echo Agent](agents/echo-agent.md) — Minimal reference implementation
- [Deep Agent](agents/deep-agent.md) — deepagents framework baseline
- [R0 Agent](agents/r0-agent.md) — Full-featured general-purpose agent
- [Coding Agent](agents/coding-agent.md) — R0 + coding-profile overlays
- [Graph Builder](agents/graph-builder.md) — Shared builder factories (`core.graph_builder`)

### Integrations

- [FastAPI](integrations/fastapi.md) — REST API router factory with 15+ endpoints

### CLI & Tooling

- [CLI Reference](cli/reference.md) — `new` and `list` commands for agent scaffolding

### Evaluation Framework

- [Overview](evals/overview.md) — Rule-based and model-graded agent evaluation

### API Reference

- [Public API](api-reference/public-api.md) — Top-level exports and quick reference
- [Store Namespaces](api-reference/store-namespaces.md) — Complete namespace inventory
- [SSE Event Types](api-reference/sse-events.md) — All streaming event types and formats
