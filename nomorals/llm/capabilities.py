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

from .defaults import is_owner_model

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
    ``owner`` marks the operator's *own* models (their fine-tunes, their
    local brain) — the broker prefers these whenever they can serve, per the
    standing operator preference.  It is a routing preference, not a
    capability: an owner card that cannot serve is still never selected, and
    an owner card in cooldown still fails over.
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
    owner: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Routing tags (rinbarpen-style): "fast", "long-context", "code",
    #: "vision", "owner", "local", "reasoning", "cheap", "quality".
    #: Auto-derived in :meth:`from_provider`; operators may set explicitly.
    tags: frozenset[str] = frozenset()
    #: OpenRouter-shaped per-1M-token pricing (input, output).  When both
    #: are 0, ``cost_per_1k`` is the fallback for legacy callers.
    price_in: float = 0.0
    price_out: float = 0.0
    #: Quality prior 0..1 for complexity-tier selection (nexus
    #: ``complexity-tier``): the cheapest card whose quality meets the
    #: target wins.  Live benchmark scores override this prior.
    quality: float = 0.5
    #: Measured tokens/sec (best-effort, from benchmarks); 0 = unknown.
    throughput_tps: float = 0.0

    def __post_init__(self) -> None:
        self.capabilities = {c if isinstance(c, Capability) else capability_from(c)
                             for c in self.capabilities}
        if isinstance(self.tags, (list, set, tuple)):
            self.tags = frozenset(str(t).lower() for t in self.tags)
        self.quality = max(0.0, min(1.0, float(self.quality or 0.0)))

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

    def has_tags(self, tags: Any) -> bool:
        """True when the card carries every requested tag."""
        if isinstance(tags, str):
            tags = {tags}
        wanted = {str(t).lower() for t in (tags or ())}
        return wanted <= set(self.tags)

    @property
    def price_per_1m_in(self) -> float:
        """USD per 1M input tokens (OpenRouter shape)."""
        if self.price_in > 0:
            return self.price_in
        return self.cost_per_1k * 1000.0

    @property
    def price_per_1m_out(self) -> float:
        if self.price_out > 0:
            return self.price_out
        return self.cost_per_1k * 1000.0

    def estimated_call_cost(self, prompt_tokens: int = 0,
                            completion_tokens: int = 1000) -> float:
        """USD estimate for one call of the given size."""
        return round(
            max(0, prompt_tokens) / 1_000_000 * self.price_per_1m_in
            + max(0, completion_tokens) / 1_000_000 * self.price_per_1m_out,
            9,
        )

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
            "owner": self.owner,
            "tags": sorted(self.tags),
            "price_in_per_1m": self.price_in,
            "price_out_per_1m": self.price_out,
            "quality": round(self.quality, 3),
            "throughput_tps": round(self.throughput_tps, 1),
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
        owner: bool = False,
        tags: Iterable[str] | None = None,
    ) -> "ModelCard":
        """Build a card from a live provider, deriving capabilities from its
        capability tokens plus code-tuned family hints.

        ``owner`` marks the operator's own models; when not given
        explicitly it is derived from the model id (any provider serving
        the operator's fine-tune counts as theirs).
        """
        tokens: set[str] = set(getattr(provider, "capabilities", set()) or set())
        name = getattr(provider, "name", "") or ""
        model_id = getattr(provider, "model_id", "") or ""
        if not owner:
            owner = is_owner_model(model_id)
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
        derived: set[str] = {str(t).lower() for t in (tags or ())}
        if local:
            derived.add("local")
        if owner:
            derived.add("owner")
        if Capability.CODE in caps:
            derived.add("code")
        if Capability.VISION in caps:
            derived.add("vision")
        if Capability.JUDGE in caps:
            derived.add("judge")
        if context_len >= 32768:
            derived.add("long-context")
        if cost_per_1k <= 0:
            derived.add("cheap")
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
            owner=owner,
            tags=frozenset(derived),
        )


def provider_capabilities() -> dict[str, frozenset[str]]:
    """The capability → provider-token map, for introspection/tests."""
    return dict(CAPABILITY_PROVIDER_TOKENS)
