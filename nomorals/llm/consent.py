"""Provider binding for the consent ledger (Echelon Wave 4).

The ledger itself (:mod:`nomorals.core.consent`) is destination-agnostic
data.  This module binds it to the provider registry:

* a provider whose spec says ``local`` (mock, ollama, llama_cpp, ocr)
  runs on the operator's own machine — data sent to it never leaves
  the machine, so no egress grant is required and the decision says
  exactly that;
* every other (hosted) provider may receive a data kind only under a
  recorded grant in the ledger, keyed by the provider's canonical id;
* an unknown provider id is a denial, never a pass — consent cannot
  be granted to a destination the registry doesn't know.

:func:`guarded_chat` is the wired call path: it runs the check and
only then calls the provider, so a caller that wants the gate gets it
in one line, and a denied call raises ``CapabilityDenied``
(code ``policy.denied``) before any byte reaches the provider.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..core.consent import ConsentLedger
from ..core.errors import CapabilityDenied
from ..core.policy import PolicyDecision
from .base import LLMProvider, LLMResponse, Message, SamplingParams
from .providers.registry import ProviderSpec, get_provider_spec

__all__ = [
    "guarded_chat",
    "provider_consent_decision",
    "require_provider_consent",
]


def _resolve(provider: LLMProvider | ProviderSpec | str) -> ProviderSpec | None:
    if isinstance(provider, ProviderSpec):
        return provider
    if isinstance(provider, LLMProvider):
        return get_provider_spec(provider.name)
    return get_provider_spec(provider)


def provider_consent_decision(
    ledger: ConsentLedger,
    provider: LLMProvider | ProviderSpec | str,
    *,
    data_kind: str,
    app: str = "devon",
    connection: str = "default",
    capability: str = "llm.chat",
) -> PolicyDecision:
    """May this provider receive ``data_kind``?  A policy decision."""
    spec = _resolve(provider)
    if spec is None:
        name = getattr(provider, "name", provider)
        return PolicyDecision(
            allowed=False,
            reason=f"unknown provider {name!r}: no registry spec, no consent possible",
            capability=capability, actor=app,
        )
    if spec.local:
        return PolicyDecision(
            allowed=True,
            reason=f"local provider {spec.id!r}: data never leaves the machine",
            capability=capability, actor=app,
        )
    return ledger.check(
        destination=spec.id, data_kind=data_kind,
        app=app, connection=connection, capability=capability,
    )


def require_provider_consent(
    ledger: ConsentLedger,
    provider: LLMProvider | ProviderSpec | str,
    *,
    data_kind: str,
    app: str = "devon",
    connection: str = "default",
    capability: str = "llm.chat",
) -> None:
    """Raise ``CapabilityDenied`` (``policy.denied``) unless consented."""
    decision = provider_consent_decision(
        ledger, provider, data_kind=data_kind,
        app=app, connection=connection, capability=capability,
    )
    if not decision.allowed:
        spec = _resolve(provider)
        raise CapabilityDenied(
            decision.reason,
            capability=capability, actor=app,
            details={
                "destination": spec.id if spec is not None else "",
                "data_kind": data_kind,
                "app": app,
                "connection": connection,
            },
        )


def guarded_chat(
    ledger: ConsentLedger,
    provider: LLMProvider,
    messages: Sequence[Message],
    *,
    data_kind: str = "chat_history",
    params: SamplingParams | None = None,
    app: str = "devon",
    connection: str = "default",
    capability: str = "llm.chat",
    **kw: Any,
) -> LLMResponse:
    """Consent-check, then chat.  Denied ⇒ ``CapabilityDenied`` raised
    before the provider is touched."""
    require_provider_consent(
        ledger, provider, data_kind=data_kind,
        app=app, connection=connection, capability=capability,
    )
    return provider.chat(messages, params, **kw)
