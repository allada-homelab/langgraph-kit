"""Context management: pressure detection + mitigation choice.

What this shows
---------------
- :class:`PressureMonitor` estimates token pressure from a message list
- The same monitor chooses between ``NONE``, ``MICROCOMPACT``,
  ``FULL_COMPACTION``, and ``STOP`` based on pressure thresholds and a
  circuit breaker for repeated compaction failures
- The kit's defaults: 70% → microcompact, 85% → full compaction

No LLM. The bundled ``PressureMiddleware`` consumes these signals to
drive automatic mitigation during a real run; this demo just shows the
detection layer.

How to run
----------
    uv run python -m examples.context_compaction

Expected output
---------------
    Tiny conversation:
      pressure=0.00 strategy=none
    After bloating with 4 large tool outputs:
      pressure=0.25 large_outputs=4 strategy=microcompact
    After pushing past the critical threshold:
      pressure=1.00 large_outputs=5 strategy=microcompact
"""

from __future__ import annotations

from examples._lib import banner, line


def main() -> None:
    banner("context_compaction")

    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        AIMessage,
        HumanMessage,
        ToolMessage,
    )

    from langgraph_kit.core.context_management.pressure import PressureMonitor

    # Tiny window so the demo can move thresholds visibly with short
    # messages. Production defaults to a 128k window. The token estimator
    # is the kit's heuristic ``len(text) // 4``; swap a real tokenizer
    # via ``token_estimator=`` when accuracy matters.
    monitor = PressureMonitor(
        window_limit=4_000,
        warn_pct=0.20,
        critical_pct=0.80,
        large_output_threshold=200,
    )

    # 1. Tiny conversation — well below warn threshold.
    tiny = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there!"),
    ]
    signals = monitor.assess(tiny)
    strategy = monitor.choose_mitigation(signals)
    line("Tiny conversation:")
    line(f"  pressure={signals.pressure_pct:.2f} strategy={strategy.value}")

    # 2. Add a few large tool outputs — crosses warn but stays below critical.
    bloated = [
        *tiny,
        *[
            ToolMessage(
                content="x" * 1000,
                tool_call_id=f"call-{i}",
                name="dummy_tool",
            )
            for i in range(4)
        ],
    ]
    signals = monitor.assess(bloated)
    strategy = monitor.choose_mitigation(signals)
    line("After bloating with 4 large tool outputs:")
    line(
        f"  pressure={signals.pressure_pct:.2f} large_outputs={signals.large_tool_outputs}"
        f" strategy={strategy.value}"
    )

    # 3. Push hard past the critical threshold.
    huge = [
        *bloated,
        AIMessage(content="y" * 16_000),
    ]
    signals = monitor.assess(huge)
    strategy = monitor.choose_mitigation(signals)
    line("After pushing past the critical threshold:")
    line(
        f"  pressure={signals.pressure_pct:.2f} large_outputs={signals.large_tool_outputs}"
        f" strategy={strategy.value}"
    )


if __name__ == "__main__":
    main()
