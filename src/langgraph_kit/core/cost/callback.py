"""LangChain callback handler for tracking token usage."""

from __future__ import annotations

import contextvars
from typing import Any

from langchain_core.callbacks import (  # pyright: ignore[reportMissingModuleSource]
    AsyncCallbackHandler,
)

from langgraph_kit.core.cost.models import TokenUsage, estimate_cost

# Context var for accumulating usage across async tasks
_usage_var: contextvars.ContextVar[list[TokenUsage]] = contextvars.ContextVar(
    "token_usage_accumulator"
)


class TokenTrackingCallback(AsyncCallbackHandler):
    """Tracks token usage from LLM calls via LangChain callbacks.

    Attach to ``config["callbacks"]`` to automatically capture token counts::

        tracker = TokenTrackingCallback()
        config["callbacks"] = [tracker]
        await graph.ainvoke(input_data, config=config)
        usage = tracker.get_accumulated()
    """

    def __init__(self) -> None:
        super().__init__()
        self._accumulated: list[TokenUsage] = []

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Extract token usage from the LLM response."""
        usage = _extract_usage(response)
        if usage is not None:
            self._accumulated.append(usage)

    def get_accumulated(self) -> list[TokenUsage]:
        """Return all accumulated token usage records."""
        return list(self._accumulated)

    def get_total(self) -> TokenUsage:
        """Return aggregated totals across all calls."""
        total_input = sum(u.input_tokens for u in self._accumulated)
        total_output = sum(u.output_tokens for u in self._accumulated)
        total_cost = sum(u.estimated_cost_usd for u in self._accumulated)
        return TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            estimated_cost_usd=total_cost,
        )

    def reset(self) -> None:
        """Clear accumulated usage."""
        self._accumulated.clear()


def _extract_usage(response: Any) -> TokenUsage | None:
    """Extract token usage from a LangChain LLM response."""
    if not hasattr(response, "llm_output") and not hasattr(response, "generations"):
        return None

    input_tokens = 0
    output_tokens = 0
    model = ""

    # Try llm_output first (OpenAI style)
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        model = llm_output.get("model_name", "") or llm_output.get("model", "")

    # Fallback: try generation_info on the first generation (Anthropic style)
    if input_tokens == 0 and hasattr(response, "generations") and response.generations:
        gen = response.generations[0]
        if isinstance(gen, list) and gen:
            gen = gen[0]
        gen_info = getattr(gen, "generation_info", None) or {}
        if isinstance(gen_info, dict):
            usage_info = gen_info.get("usage", gen_info)
            if isinstance(usage_info, dict):
                input_tokens = usage_info.get("input_tokens", 0) or usage_info.get("prompt_tokens", 0)
                output_tokens = usage_info.get("output_tokens", 0) or usage_info.get("completion_tokens", 0)
            if not model:
                model = gen_info.get("model", "")

    if input_tokens == 0 and output_tokens == 0:
        return None

    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        model=model,
    )
    usage.estimated_cost_usd = estimate_cost(usage)
    return usage
