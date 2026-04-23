"""Coverage fill — FastAPI contrib helper paths untouched by e2e.

- ``get_store`` / ``get_checkpointer`` 503 paths when contextvars aren't set.
- ``_to_lc_messages`` multipart attachment building (image/pdf/text).
- ``_extract_text_content`` shapes (str, list parts, other).
- ``_try_command`` slash-command dispatch short-circuit.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi import HTTPException

from langgraph_kit.contrib import fastapi as fastapi_mod
from langgraph_kit.contrib.fastapi import (
    _decode_data_url_text,
    _extract_text_content,
    _to_lc_messages,
    _try_command,
    get_checkpointer,
    get_store,
)
from langgraph_kit.models import ChatMessage
from langgraph_kit.models import FileAttachment as Attachment

# ---------------------------------------------------------------------------
# get_store / get_checkpointer 503 paths
# ---------------------------------------------------------------------------


def test_get_store_raises_503_when_contextvar_unset(monkeypatch: Any) -> None:
    # Force a fresh ContextVar state in the current context.
    import contextvars

    ctx = contextvars.copy_context()
    ctx.run(lambda: None)  # isolate
    # Directly assert: calling get_store in a context where the var was
    # never set raises 503 (the LookupError path).
    import contextvars as _cv

    fresh = _cv.ContextVar[Any]("fresh_test")
    monkeypatch.setattr(fastapi_mod, "_store_var", fresh)
    with pytest.raises(HTTPException) as exc:
        get_store()
    assert exc.value.status_code == 503


def test_get_checkpointer_raises_503_when_contextvar_unset(monkeypatch: Any) -> None:
    import contextvars as _cv

    fresh = _cv.ContextVar[Any]("fresh_ckpt_test")
    monkeypatch.setattr(fastapi_mod, "_checkpointer_var", fresh)
    with pytest.raises(HTTPException) as exc:
        get_checkpointer()
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# _to_lc_messages / _build_multipart_content / attachments
# ---------------------------------------------------------------------------


def _encode_text_data_url(text: str) -> str:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:text/plain;base64,{b64}"


def test_to_lc_messages_user_with_text_attachment_embeds_decoded_body() -> None:
    msg = ChatMessage(
        role="user",
        content="Look at this file:",
        attachments=[
            Attachment(
                type="text/plain",
                name="notes.txt",
                size=21,
                data_url=_encode_text_data_url("hello from attachment"),
            )
        ],
    )
    [lc] = _to_lc_messages([msg])
    parts = lc.content
    assert isinstance(parts, list)
    combined = " ".join(
        str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in parts
    )
    assert "hello from attachment" in combined
    assert "[File: notes.txt]" in combined


def test_to_lc_messages_user_with_image_attachment_uses_image_url_part() -> None:
    msg = ChatMessage(
        role="user",
        content="",
        attachments=[
            Attachment(
                type="image/png",
                name="shot.png",
                size=3,
                data_url="data:image/png;base64,AAA",
            )
        ],
    )
    [lc] = _to_lc_messages([msg])
    parts = lc.content
    assert any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in parts
    )


def test_to_lc_messages_user_with_pdf_attachment_uses_image_url_part() -> None:
    msg = ChatMessage(
        role="user",
        content="",
        attachments=[
            Attachment(
                type="application/pdf",
                name="doc.pdf",
                size=3,
                data_url="data:application/pdf;base64,PDF",
            )
        ],
    )
    [lc] = _to_lc_messages([msg])
    parts = lc.content
    # PDF takes the image_url path per the kit's current contract.
    assert any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in parts
    )


def test_to_lc_messages_assistant_and_system_pass_through() -> None:
    messages = [
        ChatMessage(role="assistant", content="prior-answer"),
        ChatMessage(role="system", content="sys"),
    ]
    lc = _to_lc_messages(messages)
    assert len(lc) == 2
    assert lc[0].content == "prior-answer"
    assert lc[1].content == "sys"


def test_decode_data_url_text_handles_malformed_input() -> None:
    # Non-base64, non-colon URL returns the raw string fallback.
    assert _decode_data_url_text("not-a-data-url") == "not-a-data-url"


def test_decode_data_url_text_decodes_base64() -> None:
    encoded = _encode_text_data_url("content!")
    assert _decode_data_url_text(encoded) == "content!"


def test_extract_text_content_string_passthrough() -> None:
    assert _extract_text_content("plain string") == "plain string"


def test_extract_text_content_list_parts_joined() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "text", "text": "last"},
        "bare-string",
    ]
    result = _extract_text_content(content)
    assert "first" in result
    assert "last" in result
    assert "[image]" in result
    assert "bare-string" in result


def test_extract_text_content_non_string_non_list_falls_back_to_str() -> None:
    assert _extract_text_content(42) == "42"


# ---------------------------------------------------------------------------
# _try_command slash-command dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_command_returns_none_for_empty_messages() -> None:
    assert await _try_command("any-agent", []) is None


@pytest.mark.asyncio
async def test_try_command_returns_none_for_non_slash_messages() -> None:
    msg = ChatMessage(role="user", content="just a regular message")
    assert await _try_command("any-agent", [msg]) is None


@pytest.mark.asyncio
async def test_try_command_returns_none_when_no_dispatcher_registered(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "langgraph_kit.contrib.fastapi.get_dispatcher", lambda _agent: None
    )
    msg = ChatMessage(role="user", content="/foo")
    assert await _try_command("any-agent", [msg]) is None


@pytest.mark.asyncio
async def test_try_command_dispatches_and_returns_sse_frame(monkeypatch: Any) -> None:
    class _FakeResult:
        handled = True
        output = "command did the thing"

    class _FakeDispatcher:
        def is_command(self, text: str) -> bool:
            _ = text
            return True

        async def dispatch(self, text: str, context: dict[str, Any]) -> Any:
            _ = text
            _ = context
            return _FakeResult()

    monkeypatch.setattr(
        "langgraph_kit.contrib.fastapi.get_dispatcher",
        lambda _agent: _FakeDispatcher(),
    )
    msg = ChatMessage(role="user", content="/echo")
    frame = await _try_command("any", [msg])
    assert frame is not None
    assert "command did the thing" in frame
    assert "[DONE]" in frame
