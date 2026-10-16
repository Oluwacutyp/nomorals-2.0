"""Fidelity: the hook that gets users in the door.

Control / judgment / language remain Devon's moat — this module does
not touch them. Fidelity is the *hook*: the output has to look and
sound good enough that a user stays to discover the moat. Three
pieces:

1. **Rent-best-model routing** — :func:`route` / :func:`pick_best_model`
   pick the best available model per task from a registry (broker
   cards, or plain ``{id, capabilities, quality}`` dicts). "Rent"
   means: use the best model available *right now* for this task —
   no loyalty to one provider, no hardcoded favourite. When the
   registry is empty, routing returns ``None`` honestly instead of
   inventing a model.
2. **Pipeline improvements as descriptors** — :class:`PostStep`
   describes ``upscale`` / ``grade`` / ``denoise`` post steps. A step
   is data until a caller injects a ``processor`` callable; without
   one, :meth:`FidelityPipeline.run` reports ``applied=False`` and
   passes the artifact through untouched — never a fake enhancement.
3. **Quality ladder** — :class:`QualityLadder` maps ``draft →
   standard → high → master`` to concrete post-step sets and minimum
   model-quality bars, so "make it better" is a ladder climb, not a
   vibe.

Universal: fidelity is per-task and per-artifact, not per-nation.
Genre/style presets live with the caller; the ladder and routing
never bake in one culture's aesthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "QualityLadder", "QUALITY_LADDER", "PostStep", "FidelityPipeline",
    "pick_best_model", "route", "POST_STEPS",
]

#: Post-step kinds the pipeline knows. Descriptors only — processors injected.
POST_STEPS = ("upscale", "grade", "denoise", "sharpen", "normalize")

#: Quality ladder: level → (post steps, min model quality 0..1).
QUALITY_LADDER: dict[str, dict[str, Any]] = {
    "draft": {"steps": [], "min_quality": 0.0,
              "blurb": "Fast sketch — no post steps."},
    "standard": {"steps": ["normalize"], "min_quality": 0.3,
                 "blurb": "Clean baseline — loudness / level normalize."},
    "high": {"steps": ["denoise", "grade", "normalize"], "min_quality": 0.6,
             "blurb": "Polished — denoise, grade, normalize."},
    "master": {"steps": ["upscale", "denoise", "grade", "sharpen", "normalize"],
               "min_quality": 0.85,
               "blurb": "Release-grade — full post chain."},
}


@dataclass(frozen=True)
class QualityLadder:
    """The quality ladder: named rungs with concrete requirements."""
    levels: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(QUALITY_LADDER))

    def rung(self, level: str) -> dict[str, Any]:
        if level not in self.levels:
            raise ValueError(f"unknown quality level {level!r}; "
                             f"known: {sorted(self.levels)}")
        return dict(self.levels[level])

    def steps_for(self, level: str) -> list[str]:
        return list(self.rung(level)["steps"])

    def climb(self, level: str) -> str | None:
        """Next rung up from ``level`` (None at the top)."""
        order = list(self.levels)
        try:
            idx = order.index(level)
        except ValueError:
            raise ValueError(f"unknown quality level {level!r}")
        return order[idx + 1] if idx + 1 < len(order) else None


# ── rent-best-model routing ─────────────────────────────────────────────────

def _card_quality(card: Any) -> float:
    if isinstance(card, dict):
        return float(card.get("quality", 0.0) or 0.0)
    for attr in ("quality", "score"):
        if hasattr(card, attr):
            try:
                return float(getattr(card, attr) or 0.0)
            except (TypeError, ValueError):
                pass
    return 0.0


def _card_caps(card: Any) -> set[str]:
    if isinstance(card, dict):
        caps = card.get("capabilities", card.get("caps", []))
        return {str(c) for c in (caps or [])}
    for attr in ("capabilities", "caps"):
        if hasattr(card, attr):
            return {str(c) for c in (getattr(card, attr) or [])}
    return set()


def _card_id(card: Any) -> str:
    if isinstance(card, dict):
        return str(card.get("id", card.get("model_id", "")))
    return str(getattr(card, "id", getattr(card, "model_id", "")))


def pick_best_model(task: str, candidates: Sequence[Any], *,
                    min_quality: float = 0.0) -> Any | None:
    """Best candidate for ``task``: serves the task, highest quality wins.

    ``candidates`` are broker cards or plain dicts with
    ``{id, capabilities, quality}``. Ties break by candidate order
    (deterministic). Returns ``None`` when nothing qualifies — the
    caller must handle that honestly instead of getting a fake model.
    """
    best: Any | None = None
    best_q = -1.0
    for card in candidates:
        if task and task not in _card_caps(card):
            continue
        q = _card_quality(card)
        if q < min_quality:
            continue
        if q > best_q:
            best, best_q = card, q
    return best


def route(task: str, registry: Any, *, min_quality: float = 0.0) -> dict[str, Any]:
    """Route ``task`` to the best model in ``registry``.

    ``registry`` may be a :class:`ModelBroker` (has ``.cards()`` /
    ``.select()``), a ``ModelRegistry`` (has ``.cards()``), or a plain
    sequence of cards/dicts. Returns ``{"model_id": ..., "quality": ...,
    "source": ...}`` or ``{"model_id": None, "reason": "no candidate"}``.
    """
    candidates: list[Any] = []
    source = "sequence"
    if hasattr(registry, "select"):
        try:
            card = registry.select(task)
        except Exception:
            card = None
        if card is not None:
            candidates = [card]
            source = "broker"
    if not candidates and hasattr(registry, "cards"):
        try:
            candidates = list(registry.cards())
            source = "registry"
        except Exception:
            candidates = []
    if not candidates and isinstance(registry, (list, tuple)):
        candidates = list(registry)
    best = pick_best_model(task, candidates, min_quality=min_quality)
    if best is None:
        return {"model_id": None, "quality": None, "source": source,
                "reason": "no candidate serves this task at the quality bar"}
    return {"model_id": _card_id(best), "quality": _card_quality(best),
            "source": source}


# ── post-step pipeline (descriptors + injected processors) ───────────────────

@dataclass
class PostStep:
    """One fidelity post step: a descriptor until a processor is injected."""
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    processor: Callable[[Any, dict[str, Any]], Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in POST_STEPS:
            raise ValueError(f"unknown post step {self.kind!r}; known: {POST_STEPS}")

    def apply(self, artifact: Any) -> dict[str, Any]:
        if self.processor is None:
            return {"applied": False, "kind": self.kind, "artifact": artifact,
                    "reason": "no processor injected — artifact passed through"}
        try:
            return {"applied": True, "kind": self.kind,
                    "artifact": self.processor(artifact, dict(self.params))}
        except Exception as exc:
            return {"applied": False, "kind": self.kind, "artifact": artifact,
                    "reason": f"{type(exc).__name__}: {exc}"}


@dataclass
class FidelityPipeline:
    """A chain of post steps at a ladder rung. Honest about what ran."""
    steps: list[PostStep] = field(default_factory=list)

    @classmethod
    def for_level(cls, level: str, *,
                  processors: dict[str, Callable[[Any, dict[str, Any]], Any]] | None = None) -> "FidelityPipeline":
        procs = processors or {}
        return cls([PostStep(kind=k, processor=procs.get(k))
                    for k in QualityLadder().steps_for(level)])

    def run(self, artifact: Any) -> dict[str, Any]:
        applied: list[str] = []
        skipped: list[str] = []
        current = artifact
        for step in self.steps:
            res = step.apply(current)
            if res["applied"]:
                applied.append(step.kind)
                current = res["artifact"]
            else:
                skipped.append(step.kind)
        return {"artifact": current, "applied": applied, "skipped": skipped,
                "steps": len(self.steps)}
