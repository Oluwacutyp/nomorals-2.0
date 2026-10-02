"""Capability vocabulary and model cards for the model broker.

The broker routes by *what a model can do* instead of by name.  Capability
values are the same lowercase tokens the providers already expose via
``LLMProvider.capabilities`` (``chat``, ``vision``, ``embed``, ``ocr``), plus
the task-level specialisations the broker reasons about:

* ``CODE``   — chat-shaped, but code-tuned (Qwen-Coder, Dolphin, codebeast …).
               Maps to the ``chat`` provider capability; the broker *prefers*
               cards flagged code-capable rather than hard-requiring it.
* ``JUDGE``  — chat-shaped, long enough context to grade other outputs.
* ``SPEECH`` — speech in/out.  No provider serves it yet; cards can advertise
               it and selection degrades gracefully to "no candidate".

A :class:`ModelCard` is the broker's unit of knowledge: one servable model,
its capabilities, its size/cost hints, and which provider serves it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["Capability", "ModelCard", "capability_from", "provider_capabilities"]


class Capability(str, enum.Enum):
    """What a model can do.  Values match the provider capability tokens."""

    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    EMBED = "embed"
    OCR = "ocr"
    SPEECH = "speech"
    JUDGE = "judge"


#: Capability → provider capability tokens that can serve it.
#: CODE and JUDGE are chat-shaped specialisations; OCR rides on vision or the
#: dedicated OCR provider token.
CAPABILITY_PROVIDER_TOKENS: dict[Capability, frozenset[str]] = {
    Capability.CHAT: frozenset({"chat"}),
    Capability.CODE: frozenset({"chat"}),
    Capability.VISION: frozenset({"vision"}),
    Capability.EMBED: frozenset({"embed"}),
    Capability.OCR: frozenset({"ocr", "vision"}),
    Capability.SPEECH: frozenset({"speech"}),
    Capability.JUDGE: frozenset({"chat"}),
}

#: Model-id fragments that mark a checkpoint as code-tuned.  Used only when
#: building cards from live providers — operator-registered cards carry their
#: capabilities explicitly.
_CODE_FAMILY_HINTS = (
    "coder", "code", "dolphin", "deepseek-coder", "starcoder", "codebeast",
    "codellama", "qwen2.5-coder", "wizardcoder",
)


def capability_from(text: str) -> Capability:
    """Parse a user-supplied capability name; raises ValueError when unknown."""
    try:
        return Capability(text.strip().lower())
    except ValueError:
        valid = ", ".join(c.value for c in Capability)
        raise ValueError(f"unknown capability {text!r}; valid: {valid}") from None


@dataclass
class ModelCard:
    """One servable model, as the broker sees it.

    ``id`` is the stable key the operator uses (``nm models use <id>``).
    ``provider`` names the router provider that serves it, and ``model_id`` is
    the provider-side model name (HF repo id, GGUF filename, …).
    """

    id: str
    capabilities: set[Capability] = field(default_factory=set)
    context_len: int = 0
    local: bool = False
    provider: str = ""
    model_id: str = ""
    quant: str = ""
    cost_per_1k: float = 0.0  #: USD per 1k tokens; 0.0 = free/local
    size_gb: float = 0.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capabilities = {c if isinstance(c, Capability) else capability_from(c)
                             for c in self.capabilities}

    @property
    def cloud(self) -> bool:
        return not self.local

    @property
    def cost_hint(self) -> str:
        if self.local:
            return "local (electricity only)"
        if self.cost_per_1k <= 0:
            return "free tier"
        return f"${self.cost_per_1k:.4f}/1k tokens"

    def serves(self, capability: Capability) -> bool:
        """True when this card can serve the capability (specialisations count)."""
        if capability in self.capabilities:
            return True
        # CODE/JUDGE are chat-shaped: any CHAT card is a fallback candidate.
        if capability in (Capability.CODE, Capability.JUDGE):
            return Capability.CHAT in self.capabilities
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capabilities": sorted(c.value for c in self.capabilities),
            "context_len": self.context_len,
            "local": self.local,
            "provider": self.provider,
            "model_id": self.model_id,
            "quant": self.quant,
            "cost_hint": self.cost_hint,
            "size_gb": self.size_gb,
            "notes": self.notes,
        }

    @classmethod
    def from_provider(
        cls,
        provider: Any,
        *,
        card_id: str = "",
        context_len: int = 0,
        local: bool = False,
        quant: str = "",
        cost_per_1k: float = 0.0,
        size_gb: float = 0.0,
        capabilities: Iterable[str | Capability] | None = None,
        notes: str = "",
    ) -> "ModelCard":
        """Build a card from a live provider, deriving capabilities from its
        capability tokens plus code-tuned family hints."""
        tokens: set[str] = set(getattr(provider, "capabilities", set()) or set())
        name = getattr(provider, "name", "") or ""
        model_id = getattr(provider, "model_id", "") or ""
        if capabilities is not None:
            caps = {c if isinstance(c, Capability) else capability_from(c)
                    for c in capabilities}
        else:
            caps = set()
            for cap in Capability:
                if CAPABILITY_PROVIDER_TOKENS[cap] & tokens:
                    caps.add(cap)
            # The OCR provider only advertises "vision"; it is OCR-first.
            if name == "ocr":
                caps.discard(Capability.VISION)
                caps.add(Capability.OCR)
            lowered = f"{model_id} {name}".lower()
            if Capability.CHAT in caps and any(h in lowered for h in _CODE_FAMILY_HINTS):
                caps.add(Capability.CODE)
            if Capability.CHAT in caps and context_len >= 8192:
                caps.add(Capability.JUDGE)
        return cls(
            id=card_id or name or model_id or "model",
            capabilities=caps,
            context_len=context_len,
            local=local,
            provider=name,
            model_id=model_id,
            quant=quant,
            cost_per_1k=cost_per_1k,
            size_gb=size_gb,
            notes=notes,
        )


def provider_capabilities() -> dict[str, frozenset[str]]:
    """The capability → provider-token map, for introspection/tests."""
    return dict(CAPABILITY_PROVIDER_TOKENS)
