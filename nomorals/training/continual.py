"""Continual learning: replay buffer + scheduled adapter-only updates.

Two invariants, both load-bearing:
1. ADAPTER-ONLY — an update touches only the adapter delta (the small
   preference/token-weight dict).  Base model weights are never opened,
   never rewritten.  A bad continual update is therefore cheap to
   discard: drop the adapter file.
2. FIXED EVAL FIRST — the eval harness is frozen before any update
   lands (fingerprint-pinned).  A candidate that scores worse on the
   fixed harness than the adapter it would replace is rejected, even
   if it fits the new batch perfectly.  The harness is the defence
   against catastrophic forgetting: it holds the old skills the new
   batch must not erase.

"Gradient-confined": the update step is a clipped, shrunk move toward
the new batch — per-token change capped (``max_step``), then pulled
back toward the previous adapter (``shrink``) — so one bad batch can
never swing the adapter far from what already worked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "ReplayBuffer",
    "FixedEvalHarness",
    "ContinualUpdater",
    "confined_update",
    "due_for_update",
]


@dataclass
class ReplayBuffer:
    """Bounded replay buffer: new examples + a reservoir of old ones.

    ``sample`` always mixes recent items with older ones (``replay_ratio``
    of the batch comes from items older than the newest ``recent_n``),
    so a scheduled update re-sees old skills while learning new ones.
    """

    capacity: int = 2000
    replay_ratio: float = 0.5
    _items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, item: Any, *, kind: str = "pair") -> None:
        self._items.append({"item": item, "kind": kind,
                            "added_at": time.time()})
        if len(self._items) > self.capacity:
            # Reservoir eviction: drop a random OLD item, never the newest.
            idx = random.randrange(0, len(self._items) - 1)
            del self._items[idx]

    def __len__(self) -> int:
        return len(self._items)

    def sample(
        self,
        batch_size: int,
        *,
        seed: int = 0,
        recent_n: int = 50,
    ) -> list[Any]:
        """A replay batch: ``replay_ratio`` old + rest recent."""
        if not self._items or batch_size <= 0:
            return []
        rng = random.Random(seed)
        n_old = int(batch_size * self.replay_ratio)
        n_old = min(n_old, max(0, len(self._items) - 1))
        old_pool = self._items[:-recent_n] if len(self._items) > recent_n else []
        recent_pool = self._items[-recent_n:]
        batch: list[Any] = []
        if old_pool and n_old:
            batch.extend(rng.sample(old_pool, min(n_old, len(old_pool))))
        remaining = batch_size - len(batch)
        if recent_pool and remaining > 0:
            batch.extend(rng.sample(
                recent_pool, min(remaining, len(recent_pool))))
        return [entry["item"] for entry in batch]

    def to_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "replay_ratio": self.replay_ratio,
                "size": len(self._items)}


class FixedEvalHarness:
    """A FROZEN eval set: pinned by fingerprint, never edited in place.

    The harness evaluates a scorer ``fn(prompt, candidate) -> float``
    on held (prompt, chosen, rejected) triples: accuracy is the fraction
    where chosen outscores rejected.  ``fingerprint()`` is the sha256 of
    the canonical eval JSON — any edit to the eval set changes it, so a
    run can prove it evaluated against the same harness it started with.
    """

    def __init__(self, eval_pairs: Sequence[Any]) -> None:
        self._pairs = list(eval_pairs)
        self._fingerprint = hashlib.sha256(
            json.dumps(
                [{"prompt": getattr(p, "prompt", ""),
                  "chosen": getattr(p, "chosen", ""),
                  "rejected": getattr(p, "rejected", "")}
                 for p in self._pairs],
                sort_keys=True).encode()).hexdigest()[:16]
        if not self._pairs:
            raise ValueError("eval harness needs at least one eval pair")

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def __len__(self) -> int:
        return len(self._pairs)

    def evaluate(self, scorer: Callable[[str, str], float]) -> dict[str, Any]:
        wins = 0.0
        for pair in self._pairs:
            s_c = scorer(pair.prompt, pair.chosen)
            s_r = scorer(pair.prompt, pair.rejected)
            wins += 1.0 if s_c > s_r else (0.5 if s_c == s_r else 0.0)
        accuracy = wins / len(self._pairs)
        return {"accuracy": round(accuracy, 4), "n": len(self._pairs),
                "fingerprint": self._fingerprint}

    def check_fingerprint(self, fingerprint: str) -> bool:
        return fingerprint == self._fingerprint


def confined_update(
    old_weights: dict[str, float],
    batch_pairs: Sequence[Any],
    *,
    lr: float = 0.5,
    max_step: float = 1.0,
    shrink: float = 0.05,
) -> dict[str, float]:
    """One gradient-confined adapter step from a replay batch.

    For each pair: chosen tokens +``lr * weight``, rejected tokens
    -``lr * weight``.  The per-token move is clipped to ``max_step``,
    and the whole vector is shrunk ``shrink`` toward the old adapter
    (elastic pull) — the confinement that keeps one batch from
    dragging the adapter away from everything it already knew.
    """
    new = dict(old_weights)
    delta: dict[str, float] = {}
    for pair in batch_pairs:
        w = float(getattr(pair, "weight", 1.0) or 1.0)
        chosen = set(_pair_tokens(getattr(pair, "chosen", "")))
        rejected = set(_pair_tokens(getattr(pair, "rejected", "")))
        for tok in chosen:
            delta[tok] = delta.get(tok, 0.0) + lr * w
        for tok in rejected:
            delta[tok] = delta.get(tok, 0.0) - lr * w
    for tok, move in delta.items():
        clipped = max(-max_step, min(max_step, move))
        new[tok] = new.get(tok, 0.0) * (1.0 - shrink) + clipped
    return new


def _pair_tokens(text: str) -> list[str]:
    import re as _re

    return _re.findall(r"[a-z0-9][a-z0-9'_-]*", (text or "").lower())


def due_for_update(last_run_at: float, *, interval_hours: float = 24.0,
                   now: float | None = None) -> bool:
    """True when the scheduled interval has elapsed since ``last_run_at``."""
    now = time.time() if now is None else now
    return (now - last_run_at) >= interval_hours * 3600.0


@dataclass
class ContinualUpdater:
    """Scheduled, light, adapter-only updates with a fixed eval harness."""

    harness: FixedEvalHarness
    buffer: ReplayBuffer = field(default_factory=ReplayBuffer)
    interval_hours: float = 24.0
    min_batch: int = 8
    last_run_at: float = 0.0

    def ingest(self, items: Sequence[Any], *, kind: str = "pair") -> int:
        for item in items:
            self.buffer.add(item, kind=kind)
        return len(items)

    def scheduled_update(
        self,
        adapter: Any,
        *,
        batch_size: int = 32,
        seed: int = 0,
        lr: float = 0.5,
        max_step: float = 1.0,
        shrink: float = 0.05,
        force: bool = False,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Run one scheduled update.  Returns the verdict + candidate.

        Steps: due-check -> replay batch -> confined adapter-only step ->
        fixed-harness eval -> reject if the harness regresses.  The
        caller's adapter is NEVER mutated; the candidate is a new
        adapter object the caller may adopt or discard.
        """
        now = time.time() if now is None else now
        if not force and not due_for_update(self.last_run_at,
                                            interval_hours=self.interval_hours,
                                            now=now):
            return {"status": "not_due", "last_run_at": self.last_run_at,
                    "interval_hours": self.interval_hours}
        if len(self.buffer) < self.min_batch:
            return {"status": "deferred", "buffer": len(self.buffer),
                    "min_batch": self.min_batch,
                    "reason": "not enough replay data yet"}
        batch = self.buffer.sample(batch_size, seed=seed)
        new_weights = confined_update(
            dict(getattr(adapter, "token_weights", {}) or {}), batch,
            lr=lr, max_step=max_step, shrink=shrink)
        import copy as _copy

        candidate = _copy.deepcopy(adapter)
        candidate.token_weights = new_weights
        candidate.pair_count = int(getattr(adapter, "pair_count", 0)) + len(batch)
        candidate.metadata = {**getattr(adapter, "metadata", {}),
                              "continual_update_at": now,
                              "continual_batch": len(batch)}
        before = self.harness.evaluate(
            lambda prompt, cand: adapter.score(prompt, cand))
        after = self.harness.evaluate(
            lambda prompt, cand: candidate.score(prompt, cand))
        self.last_run_at = now
        if after["accuracy"] < before["accuracy"]:
            _log.warning(
                "continual update REJECTED: harness %.4f -> %.4f",
                before["accuracy"], after["accuracy"])
            return {"status": "rejected",
                    "reason": "fixed-harness regression (forgetting)",
                    "before": before, "after": after,
                    "fingerprint": self.harness.fingerprint}
        return {"status": "accepted", "before": before, "after": after,
                "candidate": candidate,
                "fingerprint": self.harness.fingerprint}
