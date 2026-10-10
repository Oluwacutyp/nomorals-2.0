"""DeepSeek provider — OpenAI-compatible endpoint.

DeepSeek's API (https://api-docs.deepseek.com) is OpenAI-compatible at
``https://api.deepseek.com/v1``: the ``/v1/chat/completions`` wire format,
Bearer <redacted>, and model ids (``deepseek-chat`` = V3.x chat,
``deepseek-reasoner`` = R1 reasoning) all work through the standard
OpenAI shape. This subclasses :class:`OpenAICompatProvider` and only pins
the DeepSeek-specific parts: base URL, ``DEEPSEEK_API_KEY`` env default,
and capability set (DeepSeek serves no embeddings endpoint — advertising
``embed`` would send calls to a 404).
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compat import OpenAICompatProvider

__all__ = ["DEEPSEEK_BASE_URL", "DEEPSEEK_MODELS", "DeepSeekProvider"]


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

DEEPSEEK_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "deepseek-chat", "notes": "DeepSeek-V3.x chat; cheapest strong chat"},
    {"id": "deepseek-reasoner", "notes": "DeepSeek-R1 reasoning model"},
)


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek's chat models over their OpenAI-compatible endpoint."""

    name = "deepseek"

    def __init__(
        self,
        *,
        base_url: str = DEEPSEEK_BASE_URL,
        api_key: str = "",
        model: str = "deepseek-chat",
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("NM_DEEPSEEK_API_KEY", "")
        )
        super().__init__(
            base_url=base_url or DEEPSEEK_BASE_URL,
            api_key=key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def _fallback_models(self) -> list[str]:
        return [m["id"] for m in DEEPSEEK_MODELS]

    @property
    def capabilities(self) -> set[str]:
        # No /v1/embeddings on DeepSeek — chat/complete/vision only.
        return {"chat", "complete", "vision"}
