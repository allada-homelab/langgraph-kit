"""Runnable examples that showcase langgraph-kit features.

Hermetic by default (no API keys needed). Set
``LANGGRAPH_KIT_EXAMPLES_LLM=real`` plus ``AGENT_LLM_API_KEY`` to run any
example against a real model.

Run a single example::

    uv run python -m examples.quickstart_echo

Run the smoke suite::

    just examples-smoke
"""
