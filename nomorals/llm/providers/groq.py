"""Groq free-tier provider.

Groq serves open-weight models on its own LPU hardware at 300+ tokens/sec and
publishes an **always-free tier with no card required** (organisation-level
rate limits, verified against Groq's rate-limits docs, Aug/Sep 2026):

* ``openai/gpt-oss-120b`` / ``openai/gpt-oss-20b`` / ``qwen/qwen3.6-27b`` /
  ``qwen/qwen3.8-27b`` — 30 RPM / 1,000 requests/day / 8K TPM / 200K TPD
* ``whisper-large-v3(-turbo)`` — 20 RPM / 2,000 RPD, 8 free transcription
  hours/day (speech-to-text, not wired here)
* ``groq/compound(-mini)`` — agentic systems, 30 RPM / 250 RPD

The wire format is OpenAI-compatible (``https://api.groq.com/openai/v1``), so
this subclasses :class:`OpenAICompatProvider` and only pins the parts that
are Groq-specific: the base URL, the ``GROQ_API_KEY`` env default, and the
capability set.  Groq has **no embeddings endpoint**, so ``embed`` is dropped
from capabilities — the router then never sends embedding work to Groq, which
is exactly the failure the base class would hit at call time.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compat import OpenAICompatProvider

__all__ = ["GROQ_BASE_URL", "GROQ_FREE_MODELS", "GroqProvider"]


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

#: Free-tier chat models (Groq free plan, verified Oct 2026 against Groq's
#: published rate-limits table and live /v1/models probes).  context is the
#: vendor context window; limits are (requests/day, tokens/day) on the free
#: plan — tokens bind first for real chat exchanges (~1k tokens each).
#:
#: NOTE: the llama-3.x ids (llama-3.3-70b-versatile, llama-3.1-8b-instant)
#: that older configs used are RETIRED on Groq — they 404 with
#: model_not_found on a standard key.  If GROQ_MODEL names one, the
#: provider heals onto this roster automatically (see _fallback_models).
GROQ_FREE_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "openai/gpt-oss-120b", "context": 131072, "requests_per_day": 1000,
     "tokens_per_day": 200000, "notes": "120B open model; strongest free-tier chat"},
    {"id": "openai/gpt-oss-20b", "context": 131072, "requests_per_day": 1000,
     "tokens_per_day": 200000, "notes": "20B; cheap/fast free-tier default"},
    {"id": "qwen/qwen3.6-27b", "context": 131072, "requests_per_day": 1000,
     "tokens_per_day": 200000, "notes": "27B Qwen; strong multilingual"},
    {"id": "qwen/qwen3.8-27b", "context": 131042, "requests_per_day": 1000,
     "tokens_per_day": 200000, "notes": "27B Qwen preview"},
)


class GroqProvider(OpenAICompatProvider):
    """Groq's free tier over the OpenAI-compatible endpoint."""

    name = "groq"

    def __init__(
        self,
        *,
        base_url: str = GROQ_BASE_URL,
        api_key: str = "",
        model: str = "openai/gpt-oss-120b",
        # Groq serves at 300-1000 tok/s: a call that stalls past 60s is
        # never coming back in the interactive budget — fail over instead.
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        key = (
            api_key
            or os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("NM_GROQ_API_KEY", "")
        )
        super().__init__(
            base_url=base_url or GROQ_BASE_URL,
            api_key=key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def _fallback_models(self) -> list[str]:
        """Known-good free-tier ids, for when the configured model 404s.

        Groq retires model ids (llama-3.3-70b-versatile is gone); a stale
        GROQ_MODEL must heal onto a live id instead of 404ing every call.
        """
        return [m["id"] for m in GROQ_FREE_MODELS]

    @property
    def capabilities(self) -> set[str]:
        # Groq serves no /v1/embeddings — advertising "embed" would send
        # embedding calls to a 404. chat/complete/vision ride the chat endpoint.
        return {"chat", "complete", "vision"}

    @property
    def is_free_tier(self) -> bool:
        """True when the configured model is one of the free-plan models."""
        return self.model in {m["id"] for m in GROQ_FREE_MODELS}

    def free_model_cards(self) -> list[dict[str, Any]]:
        """The free-tier model table, for broker cards / operator UI."""
        return [dict(m) for m in GROQ_FREE_MODELS]
