"""Venice AI provider — OpenAI-compatible endpoint.

Venice (https://venice.ai) is a privacy-first, uncensored AI API:
OpenAI-compatible text, image, video and audio APIs with zero data
retention.  The text surface is a plain OpenAI wire shape at
``https://api.venice.ai/api/v1`` — ``POST /chat/completions``,
``GET /models``, ``POST /embeddings`` — with a Venice API key as the
Bearer token, so this subclasses :class:`OpenAICompatProvider` and
only pins the Venice-specific parts: base URL, ``VENICE_API_KEY``
env default, the capability set, and the model roster.

Model ids churn — Venice's own docs say to read them from
``GET /models`` and ``GET /models/traits`` at runtime and never
hardcode them from memory.  The ``VENICE_MODELS`` table below is
therefore a *walk roster* (failover order + near-miss healing), not
the live roster: every id in it was verified against Venice's live
model catalog on 2026-11-24 (docs.venice.ai/data/model-catalog.json,
generated 2026-11-24T17:20:52Z).  The live roster is
:meth:`VeniceProvider.catalog` and the live default trait is
:meth:`VeniceProvider.traits` — ``GET /models/traits?type=text``
maps ``default`` / ``default_code`` / ``most_uncensored`` to the
current model ids.

No key, no call: the defaults/settings chains skip Venice when no
key is configured, and a direct call without one raises a typed
:class:`ProviderError` naming the env var instead of sending an
unauthenticated request that can only 401.  The key is the owner's
hand (create one at venice.ai/settings/api); nothing here signs up,
spends, tops up, or stores a credential.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from ...core.errors import ProviderError, classify
from ..base import LLMResponse, Message, SamplingParams
from .openai_compat import OpenAICompatProvider, heal_model_id

__all__ = [
    "VENICE_BASE_URL",
    "VENICE_DEFAULT_MODEL",
    "VENICE_MODELS",
    "VeniceProvider",
]


_log = logging.getLogger(__name__)


VENICE_BASE_URL = "https://api.venice.ai/api/v1"

#: Venice's flagship uncensored chat model — verified live in the
#: model catalog on 2026-11-24 (primary id of the Venice Uncensored
#: 1.2 family, traits: most_uncensored, privacy: private).
VENICE_DEFAULT_MODEL = "venice-uncensored-1-2"

#: Walk roster — every id verified live in Venice's model catalog on
#: 2026-11-24.  A near-miss table + failover order, NOT the live
#: roster: models are added and retired often, so :meth:`catalog`
#: (``GET /models``) is the live source and this table is only the
#: fallback the model walk uses when the configured id 404s.
VENICE_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "venice-uncensored-1-2",
     "notes": "Venice flagship uncensored chat; most_uncensored trait"},
    {"id": "venice-uncensored-role-play",
     "notes": "Venice uncensored roleplay variant"},
    {"id": "llama-3.3-70b",
     "notes": "Llama 3.3 70B, open weights"},
    {"id": "qwen3-235b-a22b-instruct-2507",
     "notes": "Qwen3 235B instruct"},
)

#: Renamed flagship ids heal here: ``venice-uncensored`` was the
#: previous version's name for the current default model.
VENICE_ALIASES: dict[str, str] = {
    "venice-uncensored": VENICE_DEFAULT_MODEL,
}

#: What a missing key means, in one plain sentence — raised before any
#: HTTP is attempted, so a keyless call never spends a request on a
#: certain 401.  "unauthorized" is deliberate: the failure taxonomy
#: (nomorals/llm/failures.py) classifies a missing key as AUTH
#: (non-transient, park the provider) instead of UNKNOWN.
_NO_KEY_MESSAGE = (
    "venice: unauthorized — no API key configured — set VENICE_API_KEY "
    "(create one at venice.ai/settings/api) or NM_VENICE_API_KEY"
)


class VeniceProvider(OpenAICompatProvider):
    """Venice's chat models over their OpenAI-compatible endpoint."""

    name = "venice"

    def __init__(
        self,
        *,
        base_url: str = VENICE_BASE_URL,
        api_key: str = "",
        model: str = VENICE_DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        key = (
            api_key
            or os.environ.get("VENICE_API_KEY", "")
            or os.environ.get("NM_VENICE_API_KEY", "")
        )
        configured = (model or "").strip() or VENICE_DEFAULT_MODEL
        healed = VENICE_ALIASES.get(configured, configured)
        healed = heal_model_id(
            healed, {m["id"] for m in VENICE_MODELS})
        if healed != configured:
            _log.info("venice model healed: %r -> %r", model, healed)
        model = healed
        super().__init__(
            base_url=base_url or VENICE_BASE_URL,
            api_key=key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def _fallback_models(self) -> list[str]:
        """Known-good roster ids, for when the configured model 404s.

        Venice retires model ids; a stale VENICE_MODEL heals onto a
        live id instead of 404ing every call.
        """
        return [m["id"] for m in VENICE_MODELS]

    # ── fail closed without a key ──────────────────────────────────────
    def _require_key(self) -> None:
        """Raise the plain no-key error before any HTTP is attempted."""
        if not self.api_key:
            raise ProviderError(_NO_KEY_MESSAGE,
                                details={"http_status": 401})

    def health_detail(self) -> tuple[bool, str]:
        """Liveness probe — plain no-key answer instead of a 401 probe."""
        if not self.api_key:
            return False, _NO_KEY_MESSAGE
        return super().health_detail()

    def chat(
        self,
        messages: Sequence[Message],
        params: SamplingParams | None = None,
        **kw: Any,
    ) -> LLMResponse:
        self._require_key()
        return super().chat(messages, params, **kw)

    def embed(self, texts: Sequence[str], *, model: str = "",
              **kw: Any) -> list[list[float]]:
        self._require_key()
        return super().embed(texts, model=model, **kw)

    def describe_image(
        self, image: bytes, prompt: str = "",
        params: SamplingParams | None = None, **kw: Any,
    ) -> LLMResponse:
        self._require_key()
        return super().describe_image(image, prompt, params, **kw)

    # ── live discovery (the roster that never rots) ───────────────────
    def catalog(self) -> list[dict[str, Any]]:
        """The live model roster (``GET /models`` — needs the key).

        Always queried fresh: Venice adds and retires models often,
        so a cached list is a stale list.
        """
        self._require_key()
        try:
            raw = self.http.get(f"{self.base_url}/models",
                                timeout=min(30.0, self.timeout))
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(
                f"venice catalog fetch failed: {error.message}",
                retryable=error.retryable,
            ) from exc
        data = raw.json()
        models = (data or {}).get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ProviderError("venice catalog returned an unexpected shape")
        return models

    def traits(self, model_type: str = "text") -> dict[str, str]:
        """Live trait → model-id map (``GET /models/traits``).

        ``type=text`` maps ``default`` / ``default_code`` /
        ``most_uncensored`` to the current model ids — the way to ask
        Venice what its default model is today instead of trusting a
        pinned id.
        """
        self._require_key()
        try:
            raw = self.http.get(
                f"{self.base_url}/models/traits?type={model_type}",
                timeout=min(30.0, self.timeout),
            )
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(
                f"venice traits fetch failed: {error.message}",
                retryable=error.retryable,
            ) from exc
        data = raw.json()
        traits = (data or {}).get("data") if isinstance(data, dict) else None
        if not isinstance(traits, dict):
            raise ProviderError("venice traits returned an unexpected shape")
        return {str(k): str(v) for k, v in traits.items() if v}

    def default_model_id(self) -> str:
        """The live ``default`` text trait, else the pinned default."""
        try:
            return self.traits().get("default") or VENICE_DEFAULT_MODEL
        except Exception:  # noqa: BLE001 - live lookup is best-effort
            return VENICE_DEFAULT_MODEL
