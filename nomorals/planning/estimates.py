"""Honest estimates: prediction intervals that widen after misses.

Trust infrastructure for Phase 21. Every Devon time promise ships a
band, not a point: "this will take 15–25 min" instead of "20 min".

Patterns (from the logistics mine):
- Just Eat US11853909B1: quantile-style intervals that widen
  multiplicatively after misses.
- Meituan: for user-facing deadlines, quote the most conservative
  estimator — deadline = max of the estimates.

Every function never raises: a bad estimate path degrades to a wide
honest band, never a crash and never a fake-precise point.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger("nomorals.planning.estimates")

ESTIMATE_DISCLAIMER = (
    "⏱️ estimates are bands from past performance, not promises — "
    "the range widens when I've missed before."
)

# Base half-width as a fraction of the point estimate when there is no
# history: ±25% — honest about not knowing.
_BASE_HALF_WIDTH = 0.25
# Just Eat pattern: each recorded miss widens the band multiplicatively.
_MISS_WIDEN_FACTOR = 1.25
# Cap the widening so bands stay useful.
_MAX_WIDEN = 4.0
# Confidence floor/decay: every miss costs confidence, every hit restores some.
_CONF_HIT = 0.92
_CONF_MISS = 0.70


def _default_db() -> str:
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "planning", "estimates.db")


@dataclass
class Segment:
    """One modeled piece of a task: prep, travel, delivery, ..."""
    name: str
    point: float  # minutes
    low: float = 0.0
    high: float = 0.0


@dataclass
class Estimate:
    """A banded time estimate for a task."""
    task_type: str
    point: float       # minutes — best single guess
    low: float         # minutes — optimistic end of the band
    high: float        # minutes — pessimistic end of the band
    confidence: float  # 0..1
    segments: list[Segment] = field(default_factory=list)
    deadline: float = 0.0   # user-facing: the conservative quote (Meituan: max)
    source: str = "heuristic"  # learned | heuristic
    miss_count: int = 0
    #: standard deviation of the band (PERT-style; half the band span / 2
    #: by default).  Used by project-level aggregation (variances add).
    std_minutes: float = 0.0

    def format(self, start_epoch: float | None = None) -> str:
        """'ETA 4:30pm (range 4:15–4:50)' — wall-clock when a start is known."""
        lo = _fmt_band(self.low)
        hi = _fmt_band(self.high)
        pt = _fmt_band(self.point)
        head = f"⏱️ {pt} (range {lo}–{hi})"
        if start_epoch:
            eta = _fmt_clock(start_epoch + self.point * 60)
            lo_c = _fmt_clock(start_epoch + self.low * 60)
            hi_c = _fmt_clock(start_epoch + self.high * 60)
            head = f"⏱️ ETA {eta} (range {lo_c}–{hi_c})"
        conf = f" · confidence {self.confidence:.0%}"
        src = " · learned from past runs" if self.source == "learned" else ""
        return head + conf + src


def _fmt_band(minutes: float) -> str:
    try:
        m = max(0.0, float(minutes))
    except (TypeError, ValueError):
        return "? min"
    if m < 1:
        return f"{m * 60:.0f}s"
    if m < 60:
        return f"{m:.0f} min"
    h, rem = divmod(m, 60)
    return f"{h:.0f}h {rem:.0f}m" if rem >= 1 else f"{h:.0f}h"


def _fmt_clock(epoch: float) -> str:
    try:
        t = time.localtime(float(epoch))
        h = t.tm_hour % 12 or 12
        ampm = "am" if t.tm_hour < 12 else "pm"
        return f"{h}:{t.tm_min:02d}{ampm}"
    except (TypeError, ValueError):
        return "--:--"


class EstimateStore:
    """Per-task-type learned bands. SQLite. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS estimate_stats (
                       task_type TEXT PRIMARY KEY,
                       n INTEGER DEFAULT 0,
                       misses INTEGER DEFAULT 0,
                       ema_ratio REAL DEFAULT 1.0)""")
            # ema_ratio: actual/predicted EMA — the learned bias correction.
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS estimate_segments (
                       task_type TEXT, segment TEXT,
                       n INTEGER DEFAULT 0,
                       ema_minutes REAL DEFAULT 0.0,
                       ema_abs_err REAL DEFAULT 0.0,
                       PRIMARY KEY (task_type, segment))""")
            self._db.commit()
        except Exception:  # noqa: BLE001 — bad DB path → empty store
            _log.warning("estimates: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # ── estimation ────────────────────────────────────────────────

    def estimate(self, task_type: str,
                 segments: dict[str, float] | list[tuple[str, float]] | None = None,
                 ) -> Estimate:
        """Band for a task. ``segments``: prep/travel/delivery minutes each."""
        try:
            task_type = (task_type or "generic").strip().lower()[:80] or "generic"
            segs = self._norm_segments(segments)
            stats = self._stats(task_type)
            misses = stats["misses"]
            widen = min(_MAX_WIDEN, _MISS_WIDEN_FACTOR ** misses)

            seg_out: list[Segment] = []
            total_p = total_lo = total_hi = 0.0
            learned_any = False
            for name, minutes in segs:
                p, lo, hi, learned = self._segment_band(task_type, name, minutes, widen)
                seg_out.append(Segment(name, p, lo, hi))
                total_p += p
                total_lo += lo
                total_hi += hi
                learned_any = learned_any or learned

            if not seg_out:
                # No decomposition — single point, honest wide band.
                p, lo, hi, learned = self._segment_band(task_type, "total", 0.0, widen)
                seg_out = [Segment("total", p, lo, hi)]
                total_p, total_lo, total_hi = p, lo, hi
                learned_any = learned

            # Learned bias correction: if this task type historically runs
            # long, shift the whole band by the EMA ratio.
            ratio = stats["ema_ratio"]
            if stats["n"] >= 3 and ratio > 0:
                total_p *= ratio
                total_lo *= ratio
                total_hi *= ratio
                learned_any = True

            total_lo = min(total_lo, total_p)
            total_hi = max(total_hi, total_p)
            conf = self._confidence(stats)
            # Meituan: user-facing deadline = most conservative estimator.
            deadline = total_hi
            std = max(0.0, (total_hi - total_lo) / 4.0)
            return Estimate(task_type, total_p, total_lo, total_hi, conf,
                            seg_out, deadline,
                            "learned" if learned_any else "heuristic",
                            misses, std_minutes=round(std, 2))
        except Exception:  # noqa: BLE001
            _log.warning("estimates: estimate failed", exc_info=True)
            return Estimate("generic", 30.0, 15.0, 60.0, 0.3, [], 60.0,
                            "heuristic", 0, std_minutes=11.25)

    def _norm_segments(self, segments: Any) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        try:
            items = segments.items() if isinstance(segments, dict) else (segments or [])
            for item in items:
                name, minutes = item[0], item[1]
                name = str(name or "part").strip()[:40] or "part"
                try:
                    minutes = max(0.0, float(minutes))
                except (TypeError, ValueError):
                    continue
                out.append((name, minutes))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _segment_band(self, task_type: str, name: str, minutes: float,
                      widen: float) -> tuple[float, float, float, bool]:
        """One segment → (point, low, high, learned)."""
        learned = False
        point = minutes
        if self._db is not None and minutes <= 0:
            # No caller guess — use the learned segment average if we have one.
            try:
                row = self._db.execute(
                    "SELECT ema_minutes, n FROM estimate_segments "
                    "WHERE task_type = ? AND segment = ?",
                    (task_type, name)).fetchone()
                if row and row["n"] >= 2 and row["ema_minutes"] > 0:
                    point = float(row["ema_minutes"])
                    learned = True
            except Exception:  # noqa: BLE001
                pass
        # Half-width: learned absolute error when we have it, else ±25%.
        half = point * _BASE_HALF_WIDTH
        if self._db is not None:
            try:
                row = self._db.execute(
                    "SELECT ema_abs_err, n FROM estimate_segments "
                    "WHERE task_type = ? AND segment = ?",
                    (task_type, name)).fetchone()
                if row and row["n"] >= 2 and row["ema_abs_err"] > 0:
                    half = float(row["ema_abs_err"])
                    learned = True
            except Exception:  # noqa: BLE001
                pass
        half *= widen
        lo = max(0.0, point - half)
        hi = point + half
        return point, lo, hi, learned

    def _stats(self, task_type: str) -> dict:
        base = {"n": 0, "misses": 0, "ema_ratio": 1.0}
        if self._db is None:
            return base
        try:
            row = self._db.execute(
                "SELECT n, misses, ema_ratio FROM estimate_stats "
                "WHERE task_type = ?", (task_type,)).fetchone()
            if row:
                base.update(n=row["n"], misses=row["misses"],
                            ema_ratio=row["ema_ratio"] or 1.0)
        except Exception:  # noqa: BLE001
            pass
        return base

    def _confidence(self, stats: dict) -> float:
        n, misses = stats["n"], stats["misses"]
        if n == 0:
            return 0.4  # honest: no history
        hit_rate = (n - misses) / n
        # Confidence tracks hit rate, floored so a bad streak never reads 0%.
        return max(0.25, min(0.95, 0.5 + 0.5 * hit_rate))

    # ── learning from actuals ─────────────────────────────────────

    def record_actual(self, task_type: str, predicted: float, actual: float,
                      segments: dict[str, float] | None = None) -> bool:
        """Feed a real outcome in. Misses widen future bands (Just Eat)."""
        try:
            task_type = (task_type or "generic").strip().lower()[:80] or "generic"
            predicted = max(0.1, float(predicted))
            actual = max(0.0, float(actual))
            if self._db is None:
                return False
            stats = self._stats(task_type)
            # A miss = the actual landed outside the band we would have
            # quoted for the *prediction* — not a band re-fit around the actual.
            est = self.estimate(task_type, {"total": predicted})
            missed = actual > est.high or actual < est.low * 0.5
            misses = stats["misses"] + (1 if missed else 0)
            n = stats["n"] + 1
            ratio = actual / predicted
            ema = stats["ema_ratio"] * 0.8 + ratio * 0.2
            ema = min(3.0, max(0.33, ema))  # keep the bias correction sane
            self._db.execute(
                "INSERT INTO estimate_stats (task_type, n, misses, ema_ratio) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(task_type) DO UPDATE SET n=excluded.n, "
                "misses=excluded.misses, ema_ratio=excluded.ema_ratio",
                (task_type, n, misses, ema))
            # Per-segment absolute-error learning.
            for name, seg_actual in (segments or {}).items():
                try:
                    seg_actual = max(0.0, float(seg_actual))
                except (TypeError, ValueError):
                    continue
                name = str(name)[:40]
                row = self._db.execute(
                    "SELECT n, ema_minutes, ema_abs_err FROM estimate_segments "
                    "WHERE task_type = ? AND segment = ?", (task_type, name)).fetchone()
                if row:
                    nn = row["n"] + 1
                    em = row["ema_minutes"] * 0.8 + seg_actual * 0.2
                    # Error vs the segment's own EMA.
                    err = abs(seg_actual - row["ema_minutes"])
                    ea = row["ema_abs_err"] * 0.8 + err * 0.2
                    self._db.execute(
                        "UPDATE estimate_segments SET n=?, ema_minutes=?, "
                        "ema_abs_err=? WHERE task_type=? AND segment=?",
                        (nn, em, ea, task_type, name))
                else:
                    self._db.execute(
                        "INSERT INTO estimate_segments "
                        "(task_type, segment, n, ema_minutes, ema_abs_err) "
                        "VALUES (?, ?, 1, ?, 0.0)",
                        (task_type, name, seg_actual))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.warning("estimates: record_actual failed", exc_info=True)
            return False

    def miss_count(self, task_type: str) -> int:
        """How many times this task type has missed its band."""
        try:
            return self._stats((task_type or "").strip().lower())["misses"]
        except Exception:  # noqa: BLE001
            return 0

    # ── delay risk ────────────────────────────────────────────────

    def delay_risk(self, task_type: str, elapsed_minutes: float,
                   segments: Any = None) -> float:
        """P(this run misses the quoted band) given elapsed time. 0..1."""
        try:
            elapsed = max(0.0, float(elapsed_minutes))
            est = self.estimate(task_type, segments)
            span = max(1.0, est.high - est.point)
            # How far past the point estimate we are, in band-widths.
            overshoot = (elapsed - est.point) / span
            base = 1.0 / (1.0 + math.exp(-4.0 * overshoot))  # logistic
            # Blend with the historical miss rate — a task type that always
            # misses starts risky.
            stats = self._stats(est.task_type)
            hist = (stats["misses"] / stats["n"]) if stats["n"] else 0.15
            risk = 0.7 * base + 0.3 * hist
            return max(0.0, min(1.0, risk))
        except Exception:  # noqa: BLE001
            return 0.5

    def delay_alert(self, task_type: str, elapsed_minutes: float,
                    segments: Any = None,
                    start_epoch: float | None = None) -> str | None:
        """Proactive 'running late' message when risk is high, else None."""
        try:
            risk = self.delay_risk(task_type, elapsed_minutes, segments)
            if risk < 0.55:
                return None
            est = self.estimate(task_type, segments)
            new_point = max(float(elapsed_minutes),
                            est.point + (float(elapsed_minutes) - est.point) * 0.5)
            msg = (f"⏳ running late on {est.task_type} "
                   f"(miss risk {risk:.0%}) — new ETA {_fmt_band(new_point)} "
                   f"(was {_fmt_band(est.point)}).")
            if start_epoch:
                msg += f" New window: {_fmt_clock(start_epoch + new_point * 60)}."
            return msg
        except Exception:  # noqa: BLE001
            return None


# ── PERT three-point estimation (U.S. Navy Polaris, 1958) ──────────────

def pert(optimistic: float, most_likely: float, pessimistic: float
         ) -> tuple[float, float]:
    """Three-point estimate -> ``(expected_minutes, std_minutes)``.

    TE = (O + 4M + P) / 6 (the beta-distribution mean), sigma = (P - O) / 6.
    The classical input format for work with no history: a person quotes
    best-case / normal / worst-case and the math absorbs the uncertainty.
    Never raises; bad input -> (0.0, 0.0).
    """
    try:
        o = max(0.0, float(optimistic))
        m = max(0.0, float(most_likely))
        p = max(0.0, float(pessimistic))
        if p < o:
            o, p = p, o
        m = min(max(m, o), p)
        te = (o + 4.0 * m + p) / 6.0
        sigma = (p - o) / 6.0
        return round(te, 2), round(sigma, 2)
    except (TypeError, ValueError):
        return 0.0, 0.0


def pert_estimate(task_type: str, optimistic: float, most_likely: float,
                  pessimistic: float, *, db_path: str = "") -> Estimate:
    """A banded Estimate built from a PERT three-point quote.

    The band is TE ± 2σ (roughly a 95% interval), and it still flows
    through the learned store so recorded actuals keep correcting it.
    Never raises.
    """
    try:
        te, sigma = pert(optimistic, most_likely, pessimistic)
        low = max(0.0, te - 2.0 * sigma)
        high = te + 2.0 * sigma
        store = get_store(db_path)
        stats = store._stats((task_type or "generic").strip().lower())
        if stats["n"] >= 3 and stats["ema_ratio"] > 0:
            ratio = min(3.0, max(0.33, stats["ema_ratio"]))
            te, low, high = te * ratio, low * ratio, high * ratio
        return Estimate((task_type or "generic").strip().lower() or "generic",
                        round(te, 2), round(low, 2), round(high, 2),
                        confidence=0.5, segments=[], deadline=round(high, 2),
                        source="pert", miss_count=stats["misses"],
                        std_minutes=round(sigma, 2))
    except Exception:  # noqa: BLE001
        return Estimate("generic", 30.0, 15.0, 60.0, 0.3, [], 60.0,
                        "heuristic", 0, std_minutes=11.25)


def aggregate(task_type: str, parts: list[Estimate]) -> Estimate:
    """Project-level band from per-step estimates.

    Expected time = sum of step expectations; variances add in
    quadrature (independent steps): sigma_total = sqrt(sum(sigma^2)).
    Band = TE ± 2σ, deadline = the conservative end (Meituan).  Never
    raises; empty input -> a generic honest band.
    """
    try:
        parts = [p for p in (parts or []) if p is not None]
        if not parts:
            return Estimate((task_type or "project").strip().lower()
                            or "project", 30.0, 15.0, 60.0, 0.3, [], 60.0,
                            "heuristic", 0, std_minutes=11.25)
        te = sum(p.point for p in parts)
        var = sum((p.std_minutes or 0.0) ** 2 for p in parts)
        sigma = math.sqrt(var)
        low = max(0.0, te - 2.0 * sigma)
        high = te + 2.0 * sigma
        learned = any(p.source == "learned" for p in parts)
        pert_any = any(p.source == "pert" for p in parts)
        source = "learned" if learned else ("pert" if pert_any else "heuristic")
        conf = min(p.confidence for p in parts)
        misses = sum(p.miss_count for p in parts)
        return Estimate((task_type or "project").strip().lower() or "project",
                        round(te, 2), round(low, 2), round(high, 2),
                        round(conf, 2), [], round(high, 2), source, misses,
                        std_minutes=round(sigma, 2))
    except Exception:  # noqa: BLE001
        return Estimate("generic", 30.0, 15.0, 60.0, 0.3, [], 60.0,
                        "heuristic", 0, std_minutes=11.25)


# ── module-level convenience (lazy singleton) ────────────────────

_store: EstimateStore | None = None


def get_store(db_path: str = "") -> EstimateStore:
    global _store
    if _store is None or db_path:
        _store = EstimateStore(db_path)
    return _store


def estimate(task_type: str, segments: Any = None,
             db_path: str = "") -> Estimate:
    """Banded estimate for a task type. Never raises."""
    return get_store(db_path).estimate(task_type, segments)


def record_actual(task_type: str, predicted: float, actual: float,
                  segments: dict[str, float] | None = None,
                  db_path: str = "") -> bool:
    """Feed a real outcome in. Never raises."""
    return get_store(db_path).record_actual(task_type, predicted, actual, segments)


# ── chat ─────────────────────────────────────────────────────────

def _usage() -> str:
    return ("usage: /eta <task-type> [seg1=min] [seg2=min] ...\n"
            "       /eta record <task-type> <predicted-min> <actual-min>\n"
            "       /eta risk <task-type> <elapsed-min>\n"
            "       /eta stats [task-type]\n"
            "example: /eta research prep=10 read=20 write=15")


def control_eta(tail: str, context: Any = None, chat: Any = None,
                db_path: str = "") -> str:
    """Chat entry. Owner-only at dispatch. Never raises."""
    try:
        store = EstimateStore(db_path) if db_path else get_store()
        parts = (tail or "").strip().split()
        if not parts or parts[0] in ("help", "-h", "--help"):
            return _usage()
        if parts[0] == "record" and len(parts) >= 4:
            ok = store.record_actual(parts[1], float(parts[2]), float(parts[3]))
            misses = store.miss_count(parts[1])
            status = "logged" if ok else "couldn't log that"
            return f"📝 {status} — {parts[1]} now has {misses} miss(es). " + ESTIMATE_DISCLAIMER
        if parts[0] == "risk" and len(parts) >= 3:
            risk = store.delay_risk(parts[1], float(parts[2]))
            alert = store.delay_alert(parts[1], float(parts[2]))
            line = f"⏳ miss risk for {parts[1]}: {risk:.0%}"
            return line + (f"\n{alert}" if alert else "")
        if parts[0] == "stats":
            ttype = parts[1] if len(parts) > 1 else ""
            if ttype:
                s = store._stats(ttype.strip().lower())
                return (f"📊 {ttype}: {s['n']} runs, {s['misses']} misses, "
                        f"bias ×{s['ema_ratio']:.2f}. " + ESTIMATE_DISCLAIMER)
            return ("📊 per-task stats need a type: /eta stats <task-type>. "
                    + ESTIMATE_DISCLAIMER)
        # Default: estimate. Parse seg=min pairs; bare numbers → "total".
        ttype = parts[0]
        segs: dict[str, float] = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    segs[k.strip() or "part"] = float(v)
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    segs["total"] = segs.get("total", 0.0) + float(p)
                except (TypeError, ValueError):
                    continue
        est = store.estimate(ttype, segs or None)
        lines = [est.format()]
        if len(est.segments) > 1:
            for s in est.segments:
                lines.append(f"  · {s.name}: {_fmt_band(s.point)} "
                             f"({_fmt_band(s.low)}–{_fmt_band(s.high)})")
        # Meituan: the quoted deadline is the conservative end.
        lines.append(f"📅 quoted deadline: {_fmt_band(est.deadline)}")
        lines.append(ESTIMATE_DISCLAIMER)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        _log.warning("estimates: control_eta failed", exc_info=True)
        return "couldn't build that estimate. " + ESTIMATE_DISCLAIMER


def eta_text(task_type: str, segments: Any = None,
             start_epoch: float | None = None, db_path: str = "") -> str:
    """One-line 'ETA 4:30pm (range 4:15–4:50)' for scheduler/chat use."""
    try:
        return get_store(db_path).estimate(task_type, segments).format(start_epoch)
    except Exception:  # noqa: BLE001
        return "⏱️ estimate unavailable"
