"""TA evidence records + decay monitoring.

A setup that backtested well six months ago is not evidence today —
edge decays as regimes shift, and a store that silently keeps stale
setups is a store that lies. This module records evidence
(:class:`EvidenceRecord`: setup, timeframe, result, timestamp) and
scores it with exponential decay, so old evidence weighs less, and
:func:`monitor` flags setups older than a threshold as stale instead
of silently keeping them.

- :func:`decay_score(age_days)` — ``0.5 ** (age / half_life)``:
  1.0 fresh, 0.5 at one half-life, → 0 as age grows. Monotonically
  decreasing in age, by construction.
- :meth:`EvidenceStore.monitor` — returns the stale setups explicitly
  (``stale`` list + counts), never silently drops them: the caller
  decides whether to re-validate, retire, or keep them, with the
  staleness on the record.
- Persistence is JSON; every public function never raises.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "EvidenceRecord",
    "EvidenceStore",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_STALE_AFTER_DAYS",
    "decay_score",
    "is_stale",
    "monitor",
]

#: Default half-life: evidence halves in weight every 30 days.
DEFAULT_HALF_LIFE_DAYS = 30.0

#: Default staleness threshold: a setup older than 90 days is stale
#: until re-validated.
DEFAULT_STALE_AFTER_DAYS = 90.0

#: Result values an evidence record may carry.
RESULTS = ("win", "loss", "breakeven", "unknown")


@dataclass
class EvidenceRecord:
    """One piece of TA evidence: a setup observed on a timeframe with
    an outcome, timestamped at observation time."""

    setup: str = ""
    timeframe: str = ""
    result: str = "unknown"  # win | loss | breakeven | unknown
    timestamp: float = 0.0
    symbol: str = ""
    notes: str = ""
    record_id: str = ""

    def age_days(self, *, now: float | None = None) -> float:
        """Age in days (0.0 for a future / missing timestamp)."""
        try:
            ref = time.time() if now is None else float(now)
            return max(0.0, (ref - float(self.timestamp or 0.0)) / 86400.0)
        except (TypeError, ValueError):
            return 0.0

    def decay(self, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
              now: float | None = None) -> float:
        """This record's decayed weight — see :func:`decay_score`."""
        return decay_score(self.age_days(now=now),
                           half_life_days=half_life_days)

    def stale(self, *, stale_after_days: float = DEFAULT_STALE_AFTER_DAYS,
              now: float | None = None) -> bool:
        """True when this record is older than the threshold."""
        return is_stale(self.age_days(now=now),
                        stale_after_days=stale_after_days)

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "setup": self.setup,
                "timeframe": self.timeframe, "result": self.result,
                "timestamp": self.timestamp, "symbol": self.symbol,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRecord":
        data = data or {}
        try:
            ts = float(data.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        return cls(setup=str(data.get("setup") or ""),
                   timeframe=str(data.get("timeframe") or ""),
                   result=str(data.get("result") or "unknown"),
                   timestamp=ts, symbol=str(data.get("symbol") or ""),
                   notes=str(data.get("notes") or ""),
                   record_id=str(data.get("record_id") or ""))


def decay_score(age_days: float, *,
                half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Exponential decay weight for evidence of a given age.

    ``0.5 ** (age / half_life)`` — 1.0 at age 0, 0.5 at one half-life,
    monotonically decreasing in age, never negative. Invalid inputs
    (negative age, non-positive half-life) return 0.0 / fall back to
    the default half-life rather than raising.
    """
    try:
        age = max(0.0, float(age_days))
        half_life = float(half_life_days)
        if half_life <= 0:
            half_life = DEFAULT_HALF_LIFE_DAYS
        return round(0.5 ** (age / half_life), 6)
    except (TypeError, ValueError):
        return 0.0


def is_stale(age_days: float, *,
             stale_after_days: float = DEFAULT_STALE_AFTER_DAYS) -> bool:
    """True when ``age_days`` exceeds the staleness threshold."""
    try:
        return float(age_days) > float(stale_after_days)
    except (TypeError, ValueError):
        return False


class EvidenceStore:
    """JSON-persisted evidence records + decay monitoring."""

    def __init__(self, path: str = "") -> None:
        self._path = path or os.path.join(
            os.path.expanduser("~"), ".nomorals", "ta",
            "evidence.json")
        self._records: dict[str, EvidenceRecord] = {}
        self._load()

    # ── persistence ──

    def _load(self) -> None:
        try:
            if not os.path.isfile(self._path):
                return
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for rid, raw in data.items():
                    if isinstance(raw, dict):
                        rec = EvidenceRecord.from_dict(raw)
                        if rec.record_id:
                            self._records[rec.record_id] = rec
        except (OSError, ValueError):
            _log.debug("ta evidence load failed", exc_info=True)

    def _save(self) -> bool:
        try:
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump({rid: r.to_dict()
                           for rid, r in self._records.items()},
                          fh, sort_keys=True)
            return True
        except OSError:
            _log.debug("ta evidence save failed", exc_info=True)
            return False

    # ── records ──

    def record(self, setup: str, timeframe: str, *,
               result: str = "unknown", symbol: str = "",
               notes: str = "", timestamp: float | None = None
               ) -> EvidenceRecord | None:
        """Append an evidence record. Returns None when ``setup`` or
        ``timeframe`` is empty (never raises)."""
        try:
            setup = (setup or "").strip()
            timeframe = (timeframe or "").strip()
            if not setup or not timeframe:
                return None
            result = (result or "unknown").strip().lower()
            if result not in RESULTS:
                result = "unknown"
            rec = EvidenceRecord(
                setup=setup[:300], timeframe=timeframe[:40],
                result=result, timestamp=(
                    float(timestamp) if timestamp is not None
                    else time.time()),
                symbol=(symbol or "").strip()[:40],
                notes=(notes or "").strip()[:500],
                record_id="ev_" + uuid.uuid4().hex[:10])
            self._records[rec.record_id] = rec
            self._save()
            return rec
        except Exception:  # noqa: BLE001
            _log.debug("ta evidence record failed", exc_info=True)
            return None

    def get(self, record_id: str) -> EvidenceRecord | None:
        return self._records.get(str(record_id or ""))

    def list(self, *, setup: str = "", timeframe: str = "") -> list[EvidenceRecord]:
        """All records, newest first; optional setup / timeframe filter."""
        try:
            items = sorted(self._records.values(),
                           key=lambda r: r.timestamp, reverse=True)
            if setup:
                items = [r for r in items if r.setup == setup]
            if timeframe:
                items = [r for r in items if r.timeframe == timeframe]
            return items
        except Exception:  # noqa: BLE001
            return []

    def weighted_results(self, setup: str, *,
                         half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                         now: float | None = None) -> dict[str, Any]:
        """Decay-weighted win/loss summary for one setup.

        Each record contributes its :func:`decay_score` weight to its
        result bucket — fresh wins count more than ancient ones.
        Returns weights per result + a decayed win rate (None when no
        decisive evidence). Never raises.
        """
        try:
            weights: dict[str, float] = {"win": 0.0, "loss": 0.0,
                                         "breakeven": 0.0, "unknown": 0.0}
            for rec in self.list(setup=setup):
                weights[rec.result] = weights.get(rec.result, 0.0) + rec.decay(
                    half_life_days=half_life_days, now=now)
            decisive = weights["win"] + weights["loss"]
            win_rate = (round(weights["win"] / decisive, 4)
                        if decisive > 0 else None)
            return {"setup": setup, "weights": {
                k: round(v, 4) for k, v in weights.items()},
                "decayed_win_rate": win_rate,
                "records": len(self.list(setup=setup))}
        except Exception:  # noqa: BLE001
            return {"setup": setup, "weights": {}, "decayed_win_rate": None,
                    "records": 0}

    # ── decay monitoring ──

    def monitor(self, *,
                stale_after_days: float = DEFAULT_STALE_AFTER_DAYS,
                half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                now: float | None = None) -> dict[str, Any]:
        """Flag stale setups. Returns ``{"stale", "fresh", "total",
        "stale_setups"}`` where ``stale`` is the list of stale records
        (with age + decay weight) and ``stale_setups`` the distinct
        setup names affected. Nothing is silently dropped — stale
        records stay in the store until the caller retires them
        explicitly via :meth:`retire`. Never raises.
        """
        try:
            ref = time.time() if now is None else float(now)
            stale: list[dict[str, Any]] = []
            fresh = 0
            for rec in self.list():
                age = rec.age_days(now=ref)
                if is_stale(age, stale_after_days=stale_after_days):
                    stale.append({
                        "record_id": rec.record_id, "setup": rec.setup,
                        "timeframe": rec.timeframe, "result": rec.result,
                        "age_days": round(age, 1),
                        "decay": decay_score(age,
                                             half_life_days=half_life_days),
                        "action": "re-validate or retire — stale evidence"})
                else:
                    fresh += 1
            stale.sort(key=lambda s: s["age_days"], reverse=True)
            return {"total": len(self._records), "fresh": fresh,
                    "stale_count": len(stale), "stale": stale,
                    "stale_setups": sorted({s["setup"] for s in stale}),
                    "stale_after_days": stale_after_days}
        except Exception:  # noqa: BLE001
            _log.debug("ta evidence monitor failed", exc_info=True)
            return {"total": len(self._records), "fresh": 0,
                    "stale_count": 0, "stale": [], "stale_setups": [],
                    "stale_after_days": stale_after_days}

    def retire(self, record_id: str) -> bool:
        """Explicitly remove one record (the only way stale evidence
        leaves the store — never silently). Never raises."""
        try:
            if record_id in self._records:
                del self._records[record_id]
                self._save()
                return True
            return False
        except Exception:  # noqa: BLE001
            return False


def monitor(store: EvidenceStore | None, *,
            stale_after_days: float = DEFAULT_STALE_AFTER_DAYS,
            half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
            now: float | None = None) -> dict[str, Any]:
    """Module-level :meth:`EvidenceStore.monitor` (empty-store report
    for a None store). Never raises."""
    if store is None:
        return {"total": 0, "fresh": 0, "stale_count": 0, "stale": [],
                "stale_setups": [], "stale_after_days": stale_after_days}
    return store.monitor(stale_after_days=stale_after_days,
                         half_life_days=half_life_days, now=now)
