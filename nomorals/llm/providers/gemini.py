"""Gemini provider — Google's OpenAI-compatible endpoint.

Google documents OpenAI compatibility for the Gemini Developer API
(https://ai.google.dev/gemini-api/docs/openai): the base URL
``https://generativelanguage.googleapis.com/v1beta/openai/`` serves the
standard ``/v1/chat/completions`` shape with a normal ``Bearer
<API-key>`` header. This subclasses :class:`OpenAICompatProvider` and
only pins the Gemini-specific parts: base URL, ``GEMINI_API_KEY`` env
default, and capability set.

The same key also drives the ``google_flow`` connector (Veo video) —
one Google AI Studio key, two surfaces.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compat import OpenAICompatProvider

__all__ = ["GEMINI_BASE_URL", "GEMINI_MODELS", "GeminiProvider"]


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

GEMINI_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "gemini-2.5-flash", "notes": "fast + cheap; strong default"},
    {"id": "gemini-2.5-pro", "notes": "strongest reasoning"},
    {"id": "gemini-2.0-flash", "notes": "previous-gen fast fallback"},
)


class GeminiProvider(OpenAICompatProvider):
    """Gemini models over Google's OpenAI-compatible endpoint."""

    name = "gemini"

    def __init__(
        self,
        *,
        base_url: str = GEMINI_BASE_URL,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("NM_GEMINI_API_KEY", "")
        )
        super().__init__(
            base_url=base_url or GEMINI_BASE_URL,
            api_key=key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def _fallback_models(self) -> list[str]:
        return [m["id"] for m in GEMINI_MODELS]

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "complete", "vision", "embed"}
