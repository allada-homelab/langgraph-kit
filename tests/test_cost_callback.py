"""Coverage fill — ``TokenTrackingCallback`` usage extraction.

The callback parses token usage from LangChain LLM responses in both
OpenAI-style (``llm_output.token_usage``) and Anthropic-style
(``generations[...].generation_info.usage``) shapes. Unit tests exercise
each parse branch + accumulation/reset logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.cost.callback import TokenTrackingCallback


class _Response:
    def __init__(
        self, *, llm_output: Any = None, generations: Any = None
    ) -> None:
        self.llm_output = llm_output
        self.generations = generations


class _Gen:
    def __init__(self, generation_info: dict[str, Any]) -> None:
        self.generation_info = generation_info


@pytest.mark.asyncio
async def test_openai_style_token_usage_extracted() -> None:
    cb = TokenTrackingCallback()
    response = _Response(
        llm_output={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model_name": "gpt-4o",
        }
    )
    await cb.on_llm_end(response, run_id="r")
    usage = cb.get_accumulated()
    assert len(usage) == 1
    assert usage[0].input_tokens == 100
    assert usage[0].output_tokens == 50
    assert usage[0].model == "gpt-4o"


@pytest.mark.asyncio
async def test_anthropic_style_token_usage_extracted() -> None:
    cb = TokenTrackingCallback()
    response = _Response(
        generations=[
            [
                _Gen(
                    generation_info={
                        "usage": {"input_tokens": 200, "output_tokens": 80},
                        "model": "claude-sonnet-4",
                    }
                )
            ]
        ]
    )
    await cb.on_llm_end(response, run_id="r")
    usage = cb.get_accumulated()
    assert len(usage) == 1
    assert usage[0].input_tokens == 200
    assert usage[0].output_tokens == 80
    assert usage[0].model == "claude-sonnet-4"


@pytest.mark.asyncio
async def test_openai_usage_key_alias_works() -> None:
    """Some providers surface ``usage`` instead of ``token_usage``."""
    cb = TokenTrackingCallback()
    response = _Response(
        llm_output={"usage": {"input_tokens": 10, "output_tokens": 5}}
    )
    await cb.on_llm_end(response, run_id="r")
    assert cb.get_accumulated()[0].input_tokens == 10


@pytest.mark.asyncio
async def test_response_with_no_usage_info_is_skipped() -> None:
    cb = TokenTrackingCallback()
    response = _Response(llm_output={}, generations=[])
    await cb.on_llm_end(response, run_id="r")
    assert cb.get_accumulated() == []


@pytest.mark.asyncio
async def test_response_lacking_both_fields_returns_none() -> None:
    """A response object with neither llm_output nor generations is skipped."""

    class _Bare:
        pass

    cb = TokenTrackingCallback()
    await cb.on_llm_end(_Bare(), run_id="r")
    assert cb.get_accumulated() == []


@pytest.mark.asyncio
async def test_get_total_aggregates_across_multiple_calls() -> None:
    cb = TokenTrackingCallback()
    r1 = _Response(
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )
    r2 = _Response(
        llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 15}}
    )
    await cb.on_llm_end(r1, run_id="r1")
    await cb.on_llm_end(r2, run_id="r2")

    total = cb.get_total()
    assert total.input_tokens == 30
    assert total.output_tokens == 20
    assert total.total_tokens == 50


@pytest.mark.asyncio
async def test_reset_clears_accumulated_usage() -> None:
    cb = TokenTrackingCallback()
    await cb.on_llm_end(
        _Response(
            llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        ),
        run_id="r",
    )
    assert cb.get_accumulated()
    cb.reset()
    assert cb.get_accumulated() == []


@pytest.mark.asyncio
async def test_generation_info_bare_dict_without_usage_wrapper() -> None:
    """Some providers put usage fields directly on generation_info, no 'usage' key."""
    cb = TokenTrackingCallback()
    response = _Response(
        generations=[
            [
                _Gen(
                    generation_info={
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "model": "x",
                    }
                )
            ]
        ]
    )
    await cb.on_llm_end(response, run_id="r")
    got = cb.get_accumulated()
    assert got
    assert got[0].input_tokens == 7
