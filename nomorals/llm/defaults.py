"""Default provider chain: free and local first.

This is the additive wiring point for the new providers.  It does not replace
the settings-driven chain in ``agents/context.py`` — it is the opinionated
default any caller can use when no explicit config exists:

1. **Ollama** (native) — free, local, model management built in.
2. **llama.cpp server** (``llama_cpp``) — free, local GGUF serving.
3. **Groq free tier** — 300+ tok/s, no card, ~1k free requests/day.
4. **OpenRouter ``:free``** — $0 models, rotating roster.
5. **HF serverless** — existing fallback.

Keyed cloud providers register only when their key is set — a provider that
can only 401 at call time adds noise to the failover chain, not value.

``sync_broker_cards`` pushes the same providers into a
:class:`~nomorals.llm.broker.ModelBroker` as :class:`ModelCard`s with honest
``local`` / ``cost_per_1k`` flags, so capability routing *sees* the free and
local options instead of treating everything as generic cloud.  Combined with
``BrokerConstraints(prefer_local=True)`` (or ``local_only=True``), the broker
then routes chat to the local machine whenever a local model is up.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "CARD_HINTS",
    "ProviderSpec",
    "build_chain",
    "specs_from_env",
    "sync_broker_cards",
]


class ProviderSpec:
    """One chain entry: build_provider kind + constructor kwargs + card hints."""

    def __init__(
        self,
        name: str,
        kind: str,
        *,
        kwargs: dict[str, Any] | None = None,
        local: bool = False,
        cost_per_1k: float = 0.0,
        context_len: int = 0,
        env_key: str = "",
    ) -> None:
        self.name = name
        self.kind = kind
        self.kwargs = dict(kwargs or {})
        self.local = local
        self.cost_per_1k = cost_per_1k
        self.context_len = context_len
        self.env_key = env_key

    def available(self) -> bool:
        """Keyed providers are only usable when their key exists."""
        return not self.env_key or bool(os.environ.get(self.env_key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "local": self.local,
            "cost_per_1k": self.cost_per_1k,
            "context_len": self.context_len,
            "available": self.available(),
        }


#: Broker-card hints for the providers this module knows.  Keys are the
#: router registration names produced by :func:`specs_from_env`.
CARD_HINTS: dict[str, dict[str, Any]] = {
    "ollama": {"local": True, "cost_per_1k": 0.0},
    "llama_cpp": {"local": True, "cost_per_1k": 0.0},
    "groq": {"local": False, "cost_per_1k": 0.0, "context_len": 131072},
    "openrouter": {"local": False, "cost_per_1k": 0.0},
    "hf_serverless": {"local": False, "cost_per_1k": 0.0},
}


def specs_from_env() -> list[ProviderSpec]:
    """The recommended chain, filtered by what is actually usable here."""
    specs = [
        ProviderSpec(
            "ollama", "ollama", local=True,
            kwargs={"base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434")},
        ),
        ProviderSpec(
            "llama_cpp", "llama_cpp", local=True,
            kwargs={"base_url": os.environ.get("LLAMA_CPP_URL", "http://localhost:8080")},
        ),
        ProviderSpec(
            "groq", "groq",
            kwargs={"model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")},
            context_len=131072, env_key="GROQ_API_KEY",
        ),
        ProviderSpec(
            "openrouter", "openrouter",
            kwargs={"model": os.environ.get("OPENROUTER_MODEL", "openrouter/free")},
            env_key="OPENROUTER_API_KEY",
        ),
        ProviderSpec(
            "hf_serverless", "hf_serverless",
            kwargs={"model": os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")},
            env_key="HF_TOKEN",
        ),
    ]
    return [s for s in specs if s.available()]


def build_chain(specs: list[ProviderSpec] | None = None):
    """Build an :class:`LLMRouter` from specs (first spec is primary).

    Providers that fail to construct are skipped with a log line — one bad
    backend must not block the chain.
    """
    from ..core.logging_setup import get_logger
    from .providers import build_provider
    from .router import LLMRouter

    _log = get_logger(__name__)
    router = LLMRouter()
    for index, spec in enumerate(specs if specs is not None else specs_from_env()):
        try:
            provider = build_provider(spec.kind, **spec.kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad backend, not a dead chain
            _log.warning("could not build provider %s (%s): %s",
                         spec.name, spec.kind, exc)
            continue
        router.add(provider, primary=index == 0, name=spec.name)
    return router


def sync_broker_cards(broker: Any, router: Any,
                      hints: dict[str, dict[str, Any]] | None = None) -> list[Any]:
    """Register one :class:`ModelCard` per router provider on the broker.

    Starts from :meth:`ModelBroker.build_from_router` (capabilities derived
    from each provider's tokens) and then applies the local/cost/context hints
    so free-and-local routing actually has data to prefer.  Additive: cards
    already on the broker for other providers are kept.
    """
    hints = hints if hints is not None else CARD_HINTS
    cards = broker.build_from_router(router)
    for card in cards:
        hint = hints.get(card.provider) or {}
        if "local" in hint:
            card.local = bool(hint["local"])
        if "cost_per_1k" in hint:
            card.cost_per_1k = float(hint["cost_per_1k"])
        if hint.get("context_len") and not card.context_len:
            card.context_len = int(hint["context_len"])
        broker.register(card)
    return cards
