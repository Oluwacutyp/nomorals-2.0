"""OpenRouter free-models provider.

OpenRouter is an OpenAI-compatible gateway (``https://openrouter.ai/api/v1``)
fronting hundreds of models.  Models whose id ends in ``:free`` cost $0:
~20 requests/min, ~50 requests/day on a no-payment account, rising to
1,000/day after a one-time $10 credit purchase.  The ``:free`` roster
**rotates** — this module never hardcodes a default free model id.  Instead it
discovers the live roster from OpenRouter's public (no-auth) model catalog
and can fall back to the ``openrouter/free`` alias, which routes to whichever
free model currently has headroom.

Attribution headers: OpenRouter asks apps to identify themselves
(``HTTP-Referer`` + ``X-Title``).  They are harmless, help OpenRouter's abuse
handling, and the shared wiring in ``agents/context.py`` already sends them —
this class only fills in defaults when the caller didn't.
"""

from __future__ import annotations

import os
from typing import Any

from ...core.errors import ProviderError, classify
from .openai_compat import OpenAICompatProvider

__all__ = ["OPENROUTER_BASE_URL", "OpenRouterProvider"]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: ``openrouter/free`` routes to whichever free model currently has capacity —
#: the only free id that does not rot when the :free roster churns.
OPENROUTER_FREE_ALIAS = "openrouter/free"


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter gateway with free-model discovery and quota introspection."""

    name = "openrouter"

    def __init__(
        self,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str = "",
        model: str = OPENROUTER_FREE_ALIAS,
        timeout: float = 120.0,
        max_retries: int = 3,
        app_url: str = "https://github.com/Oluwacutyp/nomorals-2.0",
        app_title: str = "Devon",
        **kwargs: Any,
    ) -> None:
        key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("NM_OPENROUTER_API_KEY", "")
        )
        headers = dict(kwargs.pop("extra_headers", None) or {})
        headers.setdefault("HTTP-Referer", app_url)
        # X-Title is the long-standing name; X-OpenRouter-Title is the newer
        # one and both are honoured — send the classic one.
        headers.setdefault("X-Title", app_title)
        super().__init__(
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=headers,
            **kwargs,
        )
        self.app_url = app_url
        self.app_title = app_title

    # ── free-model discovery ─────────────────────────────────────────────────
    def catalog(self) -> list[dict[str, Any]]:
        """The full public model catalog (``GET /models`` needs no auth)."""
        try:
            raw = self.http.get(f"{self.base_url}/models",
                                timeout=min(30.0, self.timeout))
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(
                f"openrouter catalog fetch failed: {error.message}",
                retryable=error.retryable,
            ) from exc
        data = raw.json()
        models = (data or {}).get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ProviderError("openrouter catalog returned an unexpected shape")
        return models

    def free_models(self) -> list[dict[str, Any]]:
        """Live ``:free`` roster: id, context window, and per-1k pricing ($0).

        Always queried fresh — free ids churn monthly, so a cached list is a
        stale list.  ``openrouter/free`` stays the sane default model because
        the alias survives roster churn.
        """
        return [
            {
                "id": m.get("id", ""),
                "context_length": m.get("context_length", 0),
                "description": (m.get("description") or "")[:200],
            }
            for m in self.catalog()
            if isinstance(m.get("id"), str) and m["id"].endswith(":free")
        ]

    def key_info(self) -> dict[str, Any]:
        """Quota for this key: ``GET /key`` → limit_remaining, is_free_tier…

        This is how the operator checks the 50/day (or 1,000/day) free-model
        budget before a long run instead of discovering it via 429s.
        """
        try:
            raw = self.http.get(f"{self.base_url}/key",
                                timeout=min(30.0, self.timeout))
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(
                f"openrouter key lookup failed: {error.message}",
                retryable=error.retryable,
            ) from exc
        data = raw.json()
        info = (data or {}).get("data") if isinstance(data, dict) else None
        if not isinstance(info, dict):
            raise ProviderError("openrouter key lookup returned an unexpected shape")
        return info

    def chat(
        self,
        messages: Any,
        params: Any = None,
        *,
        provider_prefs: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Any:
        """Chat with OpenRouter-native provider routing preferences.

        ``provider_prefs`` becomes the request's ``provider`` object:
        ``order``, ``only``, ``ignore``, ``sort`` ("price"|"throughput"|
        "latency"), ``allow_fallbacks``, ``max_price``,
        ``require_parameters``, ``data_collection``.  This is routing
        *inside* OpenRouter (which endpoint serves the model) — the
        router's own ``route=`` knobs pick *which provider* gets the call.
        """
        if not provider_prefs:
            return super().chat(messages, params, **kw)
        allowed = {"order", "only", "ignore", "sort", "allow_fallbacks",
                   "max_price", "require_parameters", "data_collection"}
        prefs = {k: v for k, v in dict(provider_prefs).items() if k in allowed}
        original = self.extra_body
        try:
            self.extra_body = {**original, "provider": prefs}
            return super().chat(messages, params, **kw)
        finally:
            self.extra_body = original

    @property
    def is_free_model(self) -> bool:
        """True when the configured model costs nothing (``:free`` or alias)."""
        return self.model == OPENROUTER_FREE_ALIAS or self.model.endswith(":free")

    def _fallback_models(self) -> list[str]:
        """The ``openrouter/free`` alias, for when the configured id 404s.

        The ``:free`` roster churns monthly, so a pinned free id rots; the
        alias always routes to whichever free model currently has headroom.
        """
        return [OPENROUTER_FREE_ALIAS]
