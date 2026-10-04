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
3. **Benchmark score** — :class:`BenchmarkDB.score` (0..1, neutral 0.5),
   plus a cross-candidate latency rank: the fastest measured median wins.
4. **Trajectory success rate** — from an *optional* injected store
   (duck-typed: ``store.success_rate(task_kind, capability, model_id)``).
   Never imported at module level; never called when absent.
5. **Soft preferences** — ``task_kind`` specialisation (e.g. a code task
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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "BrokerConstraints":
        raw = raw or {}
        providers = raw.get("providers", ())
        if isinstance(providers, str):
            providers = (providers,)
        return cls(
            local_only=bool(raw.get("local_only", False)),
            cloud_only=bool(raw.get("cloud_only", False)),
            min_context=int(raw.get("min_context", 0) or 0),
            max_cost_per_1k=float(raw.get("max_cost_per_1k", -1.0)),
            providers=tuple(providers),
            prefer_local=bool(raw.get("prefer_local", False)),
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
            card = ModelCard.from_provider(provider, card_id=name)
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
                          if c.serves(cap) and self._fits(c, cons)]
        if not candidates:
            return None
        # 2. operator override: explicit primary always wins (if it qualifies)
        if self._primary_model_id:
            pinned = next((c for c in candidates if c.id == self._primary_model_id), None)
            if pinned is not None:
                return pinned
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
                     if c.serves(cap) and self._fits(c, cons)]
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
    @staticmethod
    def _fits(card: ModelCard, cons: BrokerConstraints) -> bool:
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
        return True

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

    def consult(self, router: Any, operation: str, task_kind: str = "") -> ModelCard | None:
        """Ask the broker which provider should serve ``operation`` and move
        the router's active provider there.  Best-effort: any failure leaves
        the router exactly as it was (the old name-based path keeps working).
        """
        capability = self.capability_for_operation(operation)
        if capability is None:
            return None
        try:
            card = self.select(capability, task_kind)
        except Exception as exc:  # noqa: BLE001
            _log.debug("broker consult failed: %s", exc)
            return None
        if card is None or not card.provider:
            return None
        try:
            if router.active != card.provider:
                router.set_active(card.provider)
        except Exception as exc:  # noqa: BLE001 — never break the call path
            _log.debug("broker could not activate %s: %s", card.provider, exc)
            return None
        return card
