"""Capability-based model broker.

``LLMRouter`` already fails over across providers by *name* and filters by
provider capability tokens.  The broker sits one level above: it answers
"which model should serve this *capability*?" by scoring candidate
:class:`ModelCard`s.

Scoring order (highest precedence first):

1. **Hard capability filter** — a card that cannot serve the capability is
   never selected.  ``CODE``/``JUDGE`` fall back to any ``CHAT`` card.
2. **Operator override** — ``promote(model_id)`` pins an explicit primary;
   the pinned card always wins while it can serve the capability.
3. **Owner preference** — when ``BrokerConstraints.prefer_owner`` is set
   (the default, per the operator's standing preference), selection is
   restricted to the operator's *own* models (``ModelCard.owner``) whenever
   at least one can serve.  The owner's brain is the default; everything
   else is a fallback.  Availability still rules: an owner card in
   cooldown is skipped by the router's failover, and ``prefer_owner=False``
   opts back into pure evidence ranking.
4. **Benchmark score** — :class:`BenchmarkDB.score` (0..1, neutral 0.5),
   plus a cross-candidate latency rank: the fastest measured median wins.
5. **Trajectory success rate** — from an *optional* injected store
   (duck-typed: ``store.success_rate(task_kind, capability, model_id)``).
   Never imported at module level; never called when absent.
6. **Soft preferences** — ``task_kind`` specialisation (e.g. a code task
   prefers ``CODE`` cards), local-first when the operator asks, longer
   context as a tie-breaker.

When no card can serve the capability the broker returns ``None`` and the
caller keeps the router's existing name-based chain — the broker never
reduces what the router could already do.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..core.logging_setup import get_logger
from .benchmarks import SCORE_WINDOW, BenchmarkDB
from .capabilities import Capability, ModelCard, capability_from

__all__ = ["BrokerConstraints", "ModelBroker", "NoCandidate"]

_log = get_logger(__name__)


class NoCandidate(Exception):
    """Raised by :meth:`ModelBroker.require` when nothing can serve."""


@dataclass
class BrokerConstraints:
    """Optional filters applied on top of capability matching."""

    local_only: bool = False
    cloud_only: bool = False
    min_context: int = 0
    max_cost_per_1k: float = -1.0  # < 0 → no cap
    providers: tuple[str, ...] = ()  # empty → any provider
    prefer_local: bool = False
    #: Prefer the operator's own models (ModelCard.owner) whenever one can
    #: serve.  The standing operator preference; set False to rank purely
    #: on measured evidence.
    prefer_owner: bool = True
    #: Minimum quality 0..1 (complexity-tier selection): cards below this
    #: are excluded.  Live benchmark scores override the card prior.
    min_quality: float = 0.0
    #: Tag filter (rinbarpen-style): at least one of these tags required.
    tags_any: tuple[str, ...] = ()
    #: Cap on USD per 1M input tokens (OpenRouter max_price shape).
    max_price_per_1m: float = -1.0  # < 0 → no cap

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "BrokerConstraints":
        raw = raw or {}
        providers = raw.get("providers", ())
        if isinstance(providers, str):
            providers = (providers,)
        tags = raw.get("tags_any", raw.get("tags", ()))
        if isinstance(tags, str):
            tags = (tags,)
        return cls(
            local_only=bool(raw.get("local_only", False)),
            cloud_only=bool(raw.get("cloud_only", False)),
            min_context=int(raw.get("min_context", 0) or 0),
            max_cost_per_1k=float(raw.get("max_cost_per_1k", -1.0)),
            providers=tuple(providers),
            prefer_local=bool(raw.get("prefer_local", False)),
            prefer_owner=bool(raw.get("prefer_owner", True)),
            min_quality=max(0.0, min(1.0, float(raw.get("min_quality", 0.0) or 0.0))),
            tags_any=tuple(str(t).lower() for t in tags),
            max_price_per_1m=float(raw.get("max_price_per_1m", -1.0)),
        )


# task_kind → capability it specialises.  "judge my draft" should prefer a
# JUDGE-capable card; "write a function" a CODE one.
_TASK_KIND_CAPABILITY = {
    "code": Capability.CODE,
    "judge": Capability.JUDGE,
    "vision": Capability.VISION,
    "ocr": Capability.OCR,
    "embed": Capability.EMBED,
    "speech": Capability.SPEECH,
    "chat": Capability.CHAT,
}


class ModelBroker:
    """Selects the best model card for a capability."""

    def __init__(
        self,
        *,
        benchmarks: BenchmarkDB | None = None,
        trajectories: Any | None = None,
        primary_model_id: str = "",
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._cards: dict[str, ModelCard] = {}
        self._lock = threading.RLock()
        self.benchmarks = benchmarks or BenchmarkDB()
        #: Duck-typed trajectory store (e.g. the cognition worker's store).
        #: Only ``success_rate(task_kind, capability, model_id) -> float`` is
        #: ever called, and only when a store was actually provided.
        self.trajectories = trajectories
        self._primary_model_id = primary_model_id
        self._previous_primary: str = ""
        w = dict(weights or {})
        self.w_benchmark = float(w.get("benchmark", 0.5))
        self.w_trajectory = float(w.get("trajectory", 0.3))
        self.w_specialisation = float(w.get("specialisation", 0.15))
        self.w_local = float(w.get("local", 0.05))
        #: Cross-candidate latency rank: the fastest candidate (by median
        #: measured latency) scores 1.0, the others scale down.  This is what
        #: makes a fast model beat a merely consistent-but-slow one — the
        #: per-model BenchmarkDB.score only measures a model against itself.
        self.w_latency = float(w.get("latency", 0.25))

    # ── card registry ────────────────────────────────────────────────────────
    def register(self, card: ModelCard) -> ModelCard:
        with self._lock:
            self._cards[card.id] = card
        _log.info("broker registered model card %s (%s)", card.id, card.provider)
        return card

    def unregister(self, card_id: str) -> bool:
        with self._lock:
            return self._cards.pop(card_id, None) is not None

    def card(self, card_id: str) -> ModelCard | None:
        with self._lock:
            return self._cards.get(card_id)

    def cards(self) -> list[ModelCard]:
        with self._lock:
            return list(self._cards.values())

    def build_from_router(self, router: Any) -> list[ModelCard]:
        """Derive cards from every provider on an :class:`LLMRouter`.

        Additive — existing cards are kept; providers already represented are
        refreshed in place.
        """
        built: list[ModelCard] = []
        for name in router.providers():
            provider = router.get(name)
            if provider is None:
                continue
            ctx_len = int(getattr(provider, "context_len", 0) or 0)
            card = ModelCard.from_provider(
                provider, card_id=name, context_len=ctx_len)
            self.register(card)
            built.append(card)
        return built

    # ── operator override ────────────────────────────────────────────────────
    @property
    def primary_model_id(self) -> str:
        return self._primary_model_id

    def promote(self, model_id: str) -> ModelCard:
        """Pin an explicit primary.  While pinned, it always wins selection
        for capabilities it can serve."""
        with self._lock:
            card = self._cards.get(model_id)
            if card is None:
                raise KeyError(f"unknown model card {model_id!r}")
            if self._primary_model_id != model_id:
                self._previous_primary = self._primary_model_id
            self._primary_model_id = model_id
        _log.info("broker primary: %s -> %s", self._previous_primary or "(none)", model_id)
        return card

    def demote(self) -> str:
        """Release the override, restoring the previous primary (may be '')."""
        with self._lock:
            self._primary_model_id, self._previous_primary = self._previous_primary, ""
            return self._primary_model_id

    # ── selection ────────────────────────────────────────────────────────────
    def select(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> ModelCard | None:
        """Best card for the capability, or ``None`` when nothing qualifies."""
        cap = capability_from(capability) if isinstance(capability, str) else capability
        cons = constraints if isinstance(constraints, BrokerConstraints) \
            else BrokerConstraints.from_mapping(constraints)
        with self._lock:
            candidates = [c for c in self._cards.values()
                          if c.serves(cap) and self._fits(c, cons, cap)]
        if not candidates:
            return None
        # 2. operator override: explicit primary always wins (if it qualifies)
        if self._primary_model_id:
            pinned = next((c for c in candidates if c.id == self._primary_model_id), None)
            if pinned is not None:
                return pinned
        # 3. owner preference: the operator's own models are the default
        # brain; everything else is a fallback.  Still dynamic — evidence
        # keeps ranking *among* the owner's models, cooldown/failover still
        # applies at serve time, and prefer_owner=False opts out.
        if cons.prefer_owner:
            owned = [c for c in candidates if c.owner]
            if owned:
                _log.debug("broker: %d owner card(s) preferred for %s/%s",
                           len(owned), cap.value, task_kind)
                candidates = owned
        special = _TASK_KIND_CAPABILITY.get((task_kind or "").lower(), None)
        # One prefetch for every candidate: score + latency rank both derive
        # from the same rows.  This used to cost three identical DB queries
        # per candidate (summary → samples, summary → score → samples, and
        # the selection score → samples again) on EVERY broker consult —
        # i.e. on every model call when a broker is wired to the router.
        bench_rows = self.benchmarks.samples_many(
            [c.id for c in candidates], cap)
        latency_rank = self._latency_rank(bench_rows)
        scored = sorted(
            ((self._score(c, cap, special, cons, latency_rank.get(c.id),
                          bench_rows.get(c.id)), c.id, c)
             for c in candidates),
            key=lambda t: (-t[0], t[1]),
        )
        winner = scored[0][2]
        _log.debug("broker select %s/%s -> %s (%.3f)",
                   cap.value, task_kind, winner.id, scored[0][0])
        return winner

    def require(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> ModelCard:
        card = self.select(capability, task_kind, constraints)
        if card is None:
            cap = capability.value if isinstance(capability, Capability) else capability
            raise NoCandidate(f"no registered model can serve capability {cap!r}")
        return card

    def ranked(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> list[tuple[ModelCard, float]]:
        """All qualifying cards with scores, best first (introspection/UI)."""
        cap = capability_from(capability) if isinstance(capability, str) else capability
        cons = constraints if isinstance(constraints, BrokerConstraints) \
            else BrokerConstraints.from_mapping(constraints)
        special = _TASK_KIND_CAPABILITY.get((task_kind or "").lower(), None)
        with self._lock:
            cands = [c for c in self._cards.values()
                     if c.serves(cap) and self._fits(c, cons, cap)]
        if cons.prefer_owner:
            owned = [c for c in cands if c.owner]
            if owned:
                cands = owned
        bench_rows = self.benchmarks.samples_many(
            [c.id for c in cands], cap)
        latency_rank = self._latency_rank(bench_rows)
        ranked = sorted(
            ((c, self._score(c, cap, special, cons, latency_rank.get(c.id),
                             bench_rows.get(c.id)))
             for c in cands),
            key=lambda t: (-t[1], t[0].id),
        )
        if self._primary_model_id:
            ranked.sort(key=lambda t: (t[0].id != self._primary_model_id, -t[1], t[0].id))
        return ranked

    # ── internals ────────────────────────────────────────────────────────────
    def _fits(self, card: ModelCard, cons: BrokerConstraints,
              capability: Capability | None = None) -> bool:
        if cons.local_only and not card.local:
            return False
        if cons.cloud_only and card.local:
            return False
        if cons.min_context and card.context_len < cons.min_context:
            return False
        if cons.max_cost_per_1k >= 0 and card.cost_per_1k > cons.max_cost_per_1k:
            return False
        if cons.providers and card.provider not in cons.providers:
            return False
        if cons.tags_any and not card.has_tags(cons.tags_any):
            return False
        if cons.max_price_per_1m >= 0 \
                and card.price_per_1m_in > cons.max_price_per_1m:
            return False
        if cons.min_quality > 0 and capability is not None \
                and self._quality_of(card, capability) < cons.min_quality:
            return False
        return True

    def _quality_of(self, card: ModelCard, capability: Capability) -> float:
        """Quality 0..1: live benchmark score overrides the card prior."""
        try:
            rows = self.benchmarks.samples(card.id, capability)
            if rows:
                return BenchmarkDB.score_rows(rows[:SCORE_WINDOW])
        except Exception:  # noqa: BLE001 — prior is the fallback
            pass
        return max(0.0, min(1.0, float(card.quality)))

    # ── tag routing / escalation / quality-tier ──────────────────────────────
    def select_by_tags(
        self,
        tags: Any,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> ModelCard | None:
        """Tag-based routing (rinbarpen style): best card carrying every tag.

        Tags like "code", "vision", "fast", "long-context", "cheap",
        "local", "owner" live on the card; the capability is inferred from
        the tags (code→CODE, vision→VISION, else CHAT).
        """
        wanted = {tags} if isinstance(tags, str) else {str(t) for t in (tags or ())}
        wanted = {t.lower() for t in wanted if t}
        if not wanted:
            return None
        cap = Capability.CHAT
        if "code" in wanted:
            cap = Capability.CODE
        elif "vision" in wanted:
            cap = Capability.VISION
        elif "embed" in wanted:
            cap = Capability.EMBED
        cons = constraints if isinstance(constraints, BrokerConstraints) \
            else BrokerConstraints.from_mapping(constraints)
        merged = BrokerConstraints(
            local_only=cons.local_only, cloud_only=cons.cloud_only,
            min_context=cons.min_context,
            max_cost_per_1k=cons.max_cost_per_1k,
            providers=cons.providers, prefer_local=cons.prefer_local,
            prefer_owner=cons.prefer_owner, min_quality=cons.min_quality,
            tags_any=tuple(sorted(wanted)),
            max_price_per_1m=cons.max_price_per_1m,
        )
        return self.select(cap, task_kind, merged)

    def escalation_chain(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> list[ModelCard]:
        """Cheapest→best ordered candidates (nexus `cascade` pattern).

        A failure escalates exactly one rung at a time instead of jumping
        to the top-quality model — minimizing expected spend on the common
        first-attempt-succeeds path, with no thresholds to tune.
        """
        ranked = self.ranked(capability, task_kind, constraints)
        # ranked is best-first; cascade wants cheapest-first among the
        # qualifying set, ordered by ascending cost then descending score.
        cards = [card for card, _ in ranked]
        cards.sort(key=lambda c: (c.price_per_1m_in, -self._quality_of(
            c, capability_from(capability)
            if isinstance(capability, str) else capability), c.id))
        return cards

    def select_for_quality(
        self,
        quality: float,
        capability: Capability | str = "chat",
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> ModelCard | None:
        """Complexity-tier selection (nexus pattern): cheapest card whose
        quality meets ``quality`` (0..1).  No thresholds to tune — the
        catalog adapts.  Falls back to the top-quality card when the
        target is unreachable.
        """
        target = max(0.0, min(1.0, float(quality)))
        cap = capability_from(capability) if isinstance(capability, str) \
            else capability
        ranked = self.ranked(cap, task_kind, constraints)
        if not ranked:
            return None
        meeting = [(card, score) for card, score in ranked
                   if self._quality_of(card, cap) >= target]
        if not meeting:
            # Target unreachable: the top-quality card, not the cheapest.
            return max(ranked,
                       key=lambda t: (self._quality_of(t[0], cap), t[1],
                                      t[0].id))[0]
        # Cheapest first among the qualifying pool.
        meeting.sort(key=lambda t: (t[0].price_per_1m_in, -t[1], t[0].id))
        return meeting[0][0]

    def explain(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> dict[str, Any]:
        """Why the winner won: full scored ranking with components.

        Operators distrust black-box picks — this shows every candidate's
        benchmark score, trajectory rate, latency rank, specialisation and
        cost, best first.
        """
        cap = capability_from(capability) if isinstance(capability, str) \
            else capability
        cons = constraints if isinstance(constraints, BrokerConstraints) \
            else BrokerConstraints.from_mapping(constraints)
        special = _TASK_KIND_CAPABILITY.get((task_kind or "").lower(), None)
        with self._lock:
            cands = [c for c in self._cards.values()
                     if c.serves(cap) and self._fits(c, cons, cap)]
        if cons.prefer_owner:
            owned = [c for c in cands if c.owner]
            if owned:
                cands = owned
        rows = self.benchmarks.samples_many([c.id for c in cands], cap)
        latency_rank = self._latency_rank(rows)
        entries: list[dict[str, Any]] = []
        for card in cands:
            card_rows = rows.get(card.id, [])
            bench = BenchmarkDB.score_rows(card_rows[:SCORE_WINDOW])
            traj = self._trajectory_rate("", cap, card.id)
            entries.append({
                "id": card.id,
                "provider": card.provider,
                "score": round(self._score(card, cap, special, cons,
                                           latency_rank.get(card.id),
                                           card_rows), 4),
                "benchmark": bench,
                "trajectory": round(traj, 4) if traj is not None else None,
                "latency_rank": (round(latency_rank[card.id], 3)
                                 if card.id in latency_rank else None),
                "specialised": special in card.capabilities
                if special is not None else False,
                "cost_per_1k": card.cost_per_1k,
                "price_per_1m_in": card.price_per_1m_in,
                "quality": round(self._quality_of(card, cap), 3),
                "tags": sorted(card.tags),
                "owner": card.owner,
                "local": card.local,
            })
        entries.sort(key=lambda e: (-e["score"], e["id"]))
        return {
            "capability": cap.value,
            "task_kind": task_kind,
            "winner": entries[0]["id"] if entries else None,
            "candidates": entries,
        }

    def format_ranking(
        self,
        capability: Capability | str,
        task_kind: str = "",
        constraints: Mapping[str, Any] | BrokerConstraints | None = None,
    ) -> str:
        """God-tier selection table for consoles and status views."""
        info = self.explain(capability, task_kind, constraints)
        cands = info["candidates"]
        if not cands:
            return f"🤖 broker: no candidate serves {info['capability']}"
        lines = [f"🤖 broker ranking — {info['capability']}"
                 + (f" / {task_kind}" if task_kind else "")]
        header = (f"  {'model':<22} {'score':>6} {'bench':>6} "
                  f"{'lat':>5} {'cost/1k':>8} {'tags'}")
        lines.append(header)
        for i, e in enumerate(cands):
            marker = "🏆" if i == 0 else "  "
            lat = (f"{e['latency_rank']:.2f}" if e["latency_rank"] is not None
                   else "—")
            tags = ",".join(e["tags"][:4]) if e["tags"] else "—"
            lines.append(
                f"{marker} {e['id']:<22} {e['score']:>6.3f} "
                f"{e['benchmark']:>6.3f} {lat:>5} "
                f"${e['cost_per_1k']:>7.4f} {tags}")
        return "\n".join(lines)

    def _trajectory_rate(self, task_kind: str, capability: Capability,
                         model_id: str) -> float | None:
        store = self.trajectories
        if store is None:
            return None
        fn = getattr(store, "success_rate", None)
        if not callable(fn):
            return None
        try:
            value = fn(task_kind or capability.value, capability.value, model_id)
        except Exception as exc:  # noqa: BLE001 — trajectory data is advisory
            _log.debug("trajectory store failed for %s: %s", model_id, exc)
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _latency_rank(
        self, rows_by_model: Mapping[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """Cross-candidate latency rank: fastest median → 1.0, others scale
        as best/median.  Models with no measurements get 0.5 (neutral).

        Takes the prefetched ``samples_many`` rows (newest first) instead of
        re-querying per model — medians match :meth:`BenchmarkDB.summary`
        exactly (same rows, same rounding).
        """
        medians: dict[str, float] = {}
        for model_id, rows in rows_by_model.items():
            latencies = [r["latency_s"] for r in rows if r["success"]]
            if latencies:
                medians[model_id] = max(
                    round(statistics.median(latencies), 4), 1e-6)
        if not medians:
            return {}
        best = min(medians.values())
        return {mid: best / lat for mid, lat in medians.items()}

    def _score(self, card: ModelCard, capability: Capability,
               special: Capability | None, cons: BrokerConstraints,
               latency_rank: float | None = None,
               bench_rows: list[dict[str, Any]] | None = None) -> float:
        if bench_rows is None:
            bench = self.benchmarks.score(card.id, capability)
        else:
            # Same value score() would return: it fetches LIMIT SCORE_WINDOW
            # newest-first rows, and this is that slice of the prefetch.
            bench = BenchmarkDB.score_rows(bench_rows[:SCORE_WINDOW])
        total = self.w_benchmark * bench
        weight_sum = self.w_benchmark
        traj = self._trajectory_rate("", capability, card.id)
        if traj is not None:
            total += self.w_trajectory * traj
            weight_sum += self.w_trajectory
        if latency_rank is not None:
            total += self.w_latency * latency_rank
            weight_sum += self.w_latency
        if special is not None and special in card.capabilities:
            total += self.w_specialisation
            weight_sum += self.w_specialisation
        if cons.prefer_local and card.local:
            total += self.w_local
            weight_sum += self.w_local
        # Longer context is a pure tie-breaker — never enough to outrank a
        # genuinely better-scoring card.
        total += min(card.context_len / 1_000_000, 0.01)
        return total / weight_sum if weight_sum else 0.0

    # ── router wiring ────────────────────────────────────────────────────────
    def capability_for_operation(self, operation: str) -> Capability | None:
        return {
            "chat": Capability.CHAT,
            "complete": Capability.CHAT,
            "vision": Capability.VISION,
            "embed": Capability.EMBED,
        }.get(operation)

    def consult(self, router: Any, operation: str, task_kind: str = "",
                constraints: Mapping[str, Any] | BrokerConstraints | None = None,
                ) -> ModelCard | None:
        """Ask the broker which provider should serve ``operation`` and move
        the router's active provider there.  Best-effort: any failure leaves
        the router exactly as it was (the old name-based path keeps working).

        Providers the router reports as cooling down are skipped — promoting
        a dead provider to active would just make the next call eat the
        failover penalty.  (A half-open breaker still admits its probe;
        ``is_cooling_down`` returns False for it.)  When every candidate is
        cooling, consult returns ``None`` and the router's failover chain
        reports the honest "all providers are cooling down" error.
        """
        capability = self.capability_for_operation(operation)
        if capability is None:
            return None
        cons = constraints if isinstance(constraints, BrokerConstraints) \
            else BrokerConstraints.from_mapping(constraints)
        try:
            ranked = self.ranked(capability, task_kind, cons)
        except Exception as exc:  # noqa: BLE001
            _log.debug("broker consult failed: %s", exc)
            return None
        cooling = getattr(router, "is_cooling_down", None)
        for card, _score in ranked:
            if not card.provider:
                continue
            if callable(cooling):
                try:
                    if cooling(card.provider):
                        continue
                except Exception:  # noqa: BLE001 — never break the call path
                    pass
            try:
                if router.active != card.provider:
                    router.set_active(card.provider)
            except Exception as exc:  # noqa: BLE001 — never break the call path
                _log.debug("broker could not activate %s: %s", card.provider, exc)
                return None
            return card
        return None
