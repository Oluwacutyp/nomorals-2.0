"""Anthropic provider — native Messages API.

Anthropic has no OpenAI-compatible chat endpoint, so this is a native
implementation of the Messages API
(https://docs.anthropic.com/en/api/messages):

* ``POST https://api.anthropic.com/v1/messages`` with headers
  ``x-api-key``, ``anthropic-version: 2023-06-01``,
  ``anthropic-dangerous-direct-browser-access`` unset (server-side use).
* Body: ``{"model", "max_tokens", "system"?, "messages": [{role, content}]}``.
  ``system`` is a top-level parameter, not a message.
* Response: ``{"content": [{"type": "text", "text": ...}], "usage":
  {"input_tokens", "output_tokens"}, "stop_reason", "model"}``.
* Vision: content blocks ``{"type": "image", "source": {"type":
  "base64", "media_type", "data"}}`` built from ``Message.images``
  (base64 data URIs or raw base64).

Roles map 1:1 except ``tool`` messages, which are folded into the
preceding user turn as text (Devon's tool protocol lives above the
provider layer; the wire format stays honest text).

Sampling: ``temperature`` and ``top_p`` pass through; ``max_tokens`` is
required by the API and defaults to 4096 (clamped from SamplingParams).
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Sequence

from ...core.errors import ModelError, ProviderError, classify
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    SamplingParams,
    Usage,
    short_error,
    validate_messages,
)

__all__ = ["ANTHROPIC_API_URL", "ANTHROPIC_VERSION", "AnthropicProvider"]

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

#: Current Claude model ids (verified against Anthropic's model docs,
#: Oct 2026). Unknown ids are passed through untouched — the API is the
#: authority on what exists.
KNOWN_MODELS = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)


class AnthropicProvider(LLMProvider):
    """Claude models over the native Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("NM_ANTHROPIC_API_KEY", "")
        )
        if not key:
            raise ProviderError(
                "anthropic: no API key — set ANTHROPIC_API_KEY "
                "(console.anthropic.com → API keys)"
            )
        self.api_key = key
        self.model = model
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            **(extra_headers or {}),
        }
        self.http = HttpClient(timeout=timeout, headers=headers)
        self._policy = BackoffPolicy(max_attempts=max(1, max_retries), cap=30.0)

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "complete", "vision"}

    def health(self) -> bool:
        # The Messages API has no cheap unauthenticated probe; a minimal
        # max_tokens call is the honest liveness check.
        try:
            resp = self.chat(
                [Message.user("ping")],
                SamplingParams(max_tokens=1, temperature=0.0),
            )
            return not resp.error
        except Exception:  # noqa: BLE001 - liveness probe
            return False

    # ── wire format ──────────────────────────────────────────────

    @staticmethod
    def _image_block(image: str) -> dict[str, Any]:
        """A base64 data URI (or raw base64) → Anthropic image block."""
        text = (image or "").strip()
        media_type = "image/jpeg"
        data = text
        if text.startswith("data:"):
            header, _, data = text.partition(",")
            mime = header[5:].split(";")[0].strip()
            if mime:
                media_type = mime
        else:
            # Raw base64 — sniff the container from magic bytes.
            try:
                raw = base64.b64decode(text[:32])
            except Exception:  # noqa: BLE001 - fall back to jpeg
                raw = b""
            if raw.startswith(b"\x89PNG"):
                media_type = "image/png"
            elif raw.startswith(b"GIF8"):
                media_type = "image/gif"
            elif raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
                media_type = "image/webp"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": data},
        }

    def _to_anthropic_messages(
        self, messages: Sequence[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "assistant" if m.role == "assistant" else "user"
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for image in m.images or []:
                blocks.append(self._image_block(image))
            if not blocks:
                continue
            # Anthropic alternates user/assistant; merge same-role runs.
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(blocks)
            else:
                turns.append({"role": role, "content": blocks})
        if not turns:
            raise ModelError("anthropic: no user/assistant turns to send")
        # The API requires the first turn to be from the user.
        if turns[0]["role"] != "user":
            turns.insert(0, {"role": "user",
                             "content": [{"type": "text",
                                          "text": "(begin)"}]})
        return ("\n\n".join(system_parts), turns)

    # ── core API ─────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[Message],
        params: SamplingParams | None = None,
        **kw: Any,
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = validate_messages(messages, who="anthropic.chat")
        sampling = (params or SamplingParams()).clamped()
        system, turns = self._to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(1, int(getattr(sampling, "max_tokens", 0)
                                     or 4096)),
            "messages": turns,
            **kw,
        }
        if system:
            payload["system"] = system
        if sampling.temperature is not None:
            payload["temperature"] = sampling.temperature
        if getattr(sampling, "top_p", None) is not None:
            payload["top_p"] = sampling.top_p
        try:
            raw = retry_call(
                lambda: self.http.post_json(ANTHROPIC_API_URL, payload),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            return self._record(
                LLMResponse(text="", model=self.model,
                            error=f"{error.code}: {short_error(exc)}",
                            raw={"exception": type(exc).__name__}),
                started,
            )
        data = raw.json()
        if not isinstance(data, dict):
            return self._record(
                LLMResponse(text="", model=self.model,
                            error="anthropic returned non-JSON",
                            raw={"body": str(data)[:500]}),
                started,
            )
        if data.get("type") == "error" or "error" in data:
            err = data.get("error") or {}
            return self._record(
                LLMResponse(
                    text="",
                    model=self.model,
                    error=f"anthropic: {err.get('type', 'error')}: "
                          f"{err.get('message', '?')}",
                    raw=data,
                ),
                started,
            )
        texts = [
            b.get("text", "")
            for b in (data.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        usage = data.get("usage") or {}
        return self._record(
            LLMResponse(
                text="".join(texts),
                model=data.get("model", self.model),
                usage=Usage(
                    prompt_tokens=int(usage.get("input_tokens") or 0),
                    completion_tokens=int(usage.get("output_tokens") or 0),
                    total_tokens=int(usage.get("input_tokens") or 0)
                    + int(usage.get("output_tokens") or 0),
                ),
                finish_reason=data.get("stop_reason", "stop"),
                raw=data,
            ),
            started,
        )

    def describe_image(
        self,
        image: bytes,
        prompt: str = "",
        params: SamplingParams | None = None,
        **kw: Any,
    ) -> LLMResponse:
        """Vision: describe raw image bytes (PNG/JPEG/GIF/WebP)."""
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image).decode(
            "ascii"
        )
        return self.chat(
            [Message.user(prompt or "Describe this image in detail.",
                          images=[data_uri])],
            params,
            **kw,
        )
