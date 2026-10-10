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
import random
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
            if path != ":memory:":
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


# ── Monte Carlo schedule risk (Beta-PERT simulation) ─────────────
#
# Mined from the schedule-risk canon (everydaybudd PERT calculator,
# timeshifted-risk-mcs, the sofka Monte Carlo skill): the deterministic
# critical path is wrong about half the time, because a near-critical path
# can overtake it. So per iteration we SAMPLE every step's Beta-PERT
# distribution and take the max over ALL paths — the simulated critical
# path — then read off P10/P50/P80/P90, per-step criticality, and a
# sensitivity (tornado) ranking. Pure stdlib: random.betavariate, no numpy.

_DEFAULT_MC_ITERATIONS = 10_000
_MAX_MC_ITERATIONS = 50_000


def pert_sample(optimistic: float, most_likely: float, pessimistic: float,
                rng: random.Random | None = None) -> float:
    """One draw from the Beta-PERT distribution for a three-point quote.

    Canonical Vose parameterization: with mean mu = (O + 4M + P) / 6,
    alpha = 1 + 4(M − O)/(P − O), beta = 1 + 4(P − M)/(P − O). Collapses to
    the constant when O == P. Never raises.
    """
    try:
        o = max(0.0, float(optimistic))
        m = max(0.0, float(most_likely))
        p = max(0.0, float(pessimistic))
        if p < o:
            o, p = p, o
        m = min(max(m, o), p)
        if p <= o:
            return o
        span = p - o
        alpha = 1.0 + 4.0 * (m - o) / span
        beta = 1.0 + 4.0 * (p - m) / span
        r = rng or random
        return o + r.betavariate(alpha, beta) * span
    except (TypeError, ValueError):
        return 0.0


@dataclass
class MonteCarloResult:
    """Outcome of a schedule-risk simulation."""
    task_type: str
    iterations: int
    seed: int
    ok: bool = True
    reason: str = ""
    p10: float = 0.0
    p50: float = 0.0
    p80: float = 0.0
    p90: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    deterministic_te: float = 0.0  # the naive sum-of-TEs critical path
    criticality: dict[str, float] = field(default_factory=dict)
    sensitivity: list[tuple[str, float]] = field(default_factory=list)
    step_labels: dict[str, str] = field(default_factory=dict)

    @property
    def contingency_p80(self) -> float:
        return max(0.0, self.p80 - self.p50)

    def format(self) -> str:
        if not self.ok:
            return f"🎲 couldn't simulate: {self.reason}"
        lines = [
            f"🎲 Monte Carlo ({self.iterations:,} iterations, seed {self.seed}):",
            f"   P50 {_fmt_band(self.p50)} · P80 {_fmt_band(self.p80)} · "
            f"P90 {_fmt_band(self.p90)}  (P10 {_fmt_band(self.p10)})",
            f"   mean {_fmt_band(self.mean)} ± {_fmt_band(self.std)} · "
            f"naive plan said {_fmt_band(self.deterministic_te)}",
            f"   💰 contingency (P80−P50): {_fmt_band(self.contingency_p80)}",
        ]
        if self.sensitivity:
            lines.append("   🔥 top variance drivers:")
            for sid, rho in self.sensitivity[:5]:
                label = self.step_labels.get(sid, sid)
                lines.append(f"      • {label} (sensitivity {rho:.2f})")
        if self.criticality:
            crit = sorted(self.criticality.items(),
                          key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("   🛤️ criticality (share of runs on the critical path):")
            for sid, frac in crit:
                label = self.step_labels.get(sid, sid)
                lines.append(f"      • {label}: {frac:.0%}")
        lines.append("Quote the P80 to stakeholders; the P90 is your downside.")
        return "\n".join(lines)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _ranks(vals: list[float]) -> list[float]:
    """Average ranks for Spearman correlation (pure Python)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


def monte_carlo(task_type: str,
                steps: list[dict[str, Any]],
                iterations: int = _DEFAULT_MC_ITERATIONS,
                seed: int | None = None) -> MonteCarloResult:
    """Simulate a project's duration distribution.

    ``steps``: list of ``{"id": str, "label": str, "optimistic": m,
    "most_likely": m, "pessimistic": m, "depends_on": [ids]}``.  Each
    iteration samples every step's Beta-PERT distribution and computes the
    project duration as the longest path through the dependency DAG — the
    *simulated* critical path, which the deterministic path misses about
    half the time.  Returns percentiles, per-step criticality index, and a
    sensitivity ranking.  Never raises.
    """
    ttype = (task_type or "project").strip().lower()[:80] or "project"
    try:
        steps = [s for s in (steps or []) if isinstance(s, dict)]
        if not steps:
            return MonteCarloResult(ttype, 0, seed or 0, ok=False,
                                    reason="no steps given")
        if len(steps) > 200:
            return MonteCarloResult(ttype, 0, seed or 0, ok=False,
                                    reason="too many steps (max 200)")
        iterations = max(100, min(_MAX_MC_ITERATIONS, int(iterations or 0)
                                  or _DEFAULT_MC_ITERATIONS))
        seed = int(seed) if seed is not None else random.randrange(2 ** 31)
        rng = random.Random(seed)

        # Normalize steps.
        ids: list[str] = []
        labels: dict[str, str] = {}
        omp: dict[str, tuple[float, float, float]] = {}
        deps: dict[str, list[str]] = {}
        for i, s in enumerate(steps):
            sid = str(s.get("id") or f"step_{i}")
            ids.append(sid)
            labels[sid] = str(s.get("label") or sid)[:60]
            try:
                o = max(0.0, float(s.get("optimistic", 0)))
                m = max(0.0, float(s.get("most_likely", 0)))
                p = max(0.0, float(s.get("pessimistic", 0)))
            except (TypeError, ValueError):
                o = m = p = 0.0
            if p < o:
                o, p = p, o
            m = min(max(m, o), p)
            omp[sid] = (o, m, p)
            dlist = s.get("depends_on") or []
            deps[sid] = [str(d) for d in dlist if str(d) in
                         {str(x.get("id") or f"step_{j}")
                          for j, x in enumerate(steps)} and str(d) != sid]

        # Topological order (Kahn); cyclic leftovers appended (honest).
        indeg = {sid: 0 for sid in ids}
        after: dict[str, list[str]] = {sid: [] for sid in ids}
        for sid in ids:
            for d in deps[sid]:
                after[d].append(sid)
                indeg[sid] += 1
        topo: list[str] = []
        ready = [sid for sid in ids if indeg[sid] == 0]
        while ready:
            sid = ready.pop(0)
            topo.append(sid)
            for nxt in after[sid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        topo += [sid for sid in ids if sid not in topo]

        # Deterministic TE critical path (the naive plan, for comparison).
        te = {sid: (o + 4 * m + p) / 6.0 for sid, (o, m, p) in omp.items()}
        det_finish: dict[str, float] = {}
        for sid in topo:
            det_finish[sid] = te[sid] + max(
                [det_finish[d] for d in deps[sid]] or [0.0])
        deterministic_te = max(det_finish.values()) if det_finish else 0.0

        # Simulate.
        totals: list[float] = []
        samples: dict[str, list[float]] = {sid: [] for sid in ids}
        crit_hits: dict[str, int] = {sid: 0 for sid in ids}
        for _ in range(iterations):
            samp = {sid: pert_sample(*omp[sid], rng=rng) for sid in ids}
            for sid in ids:
                samples[sid].append(samp[sid])
            finish: dict[str, float] = {}
            for sid in topo:
                finish[sid] = samp[sid] + max(
                    [finish[d] for d in deps[sid]] or [0.0])
            total = max(finish.values()) if finish else 0.0
            totals.append(total)
            # Criticality: backtrack from the max-finish sinks.
            on_path = {sid for sid in ids
                       if abs(finish[sid] - total) < 1e-9}
            # walk backwards: a step is critical if some critical
            # successor starts exactly when it finishes.
            changed = True
            while changed:
                changed = False
                for sid in ids:
                    if sid in on_path:
                        continue
                    for nxt in after[sid]:
                        if nxt in on_path and abs(
                                finish[sid] + samp[nxt] - finish[nxt]) < 1e-6:
                            on_path.add(sid)
                            changed = True
                            break
            for sid in on_path:
                crit_hits[sid] += 1

        totals_sorted = sorted(totals)
        total_ranks = _ranks(totals)
        sens: list[tuple[str, float]] = []
        for sid in ids:
            rho = abs(_pearson(_ranks(samples[sid]), total_ranks))
            sens.append((sid, round(rho, 3)))
        sens.sort(key=lambda kv: kv[1], reverse=True)

        mean = sum(totals) / len(totals)
        var = sum((t - mean) ** 2 for t in totals) / len(totals)
        return MonteCarloResult(
            task_type=ttype, iterations=iterations, seed=seed,
            p10=round(_percentile(totals_sorted, 10), 2),
            p50=round(_percentile(totals_sorted, 50), 2),
            p80=round(_percentile(totals_sorted, 80), 2),
            p90=round(_percentile(totals_sorted, 90), 2),
            mean=round(mean, 2), std=round(math.sqrt(var), 2),
            deterministic_te=round(deterministic_te, 2),
            criticality={sid: round(crit_hits[sid] / iterations, 3)
                         for sid in ids},
            sensitivity=sens,
            step_labels=labels,
        )
    except Exception as e:  # noqa: BLE001 - never raises
        _log.debug("monte_carlo failed", exc_info=True)
        return MonteCarloResult(ttype, 0, seed or 0, ok=False,
                                reason=str(e) or "simulation failed")


# ── Reference-class forecasting (the outside view) ────────────────
#
# Mined from Kahneman/Tversky/Flyvbjerg: the inside view (this plan's story)
# is systematically optimistic; the outside view (how tasks LIKE this
# actually turned out) is the anchor. Our EstimateStore already keeps the
# outside view (EMA bias ratio per task type) — this surfaces it explicitly
# instead of applying it silently.

def reference_class_check(task_type: str, inside_minutes: float,
                          db_path: str = "") -> dict[str, Any]:
    """Compare an inside-view quote against the reference class.

    ``inside_minutes``: what the plan says. Returns the outside-view
    anchor (learned bias ratio × quote), the empirical hit-rate, and a
    verdict. Honest when there's no history yet. Never raises.
    """
    try:
        ttype = (task_type or "generic").strip().lower()[:80] or "generic"
        inside = max(0.0, float(inside_minutes))
        store = get_store(db_path)
        stats = store._stats(ttype)
        n, misses, ratio = stats["n"], stats["misses"], stats["ema_ratio"]
        if n < 3:
            return {"ok": True, "task_type": ttype, "inside": inside,
                    "reference_class_size": n,
                    "verdict": ("no reference class yet — fewer than 3 runs "
                                "recorded. Quote wide and start recording.")}

        outside = inside * ratio
        hit_rate = (n - misses) / n if n else 0.0
        if ratio > 1.25:
            verdict = (f"chronic optimism: this task type runs ×{ratio:.2f} "
                       f"the quote. Anchor at {_fmt_band(outside)}, not "
                       f"{_fmt_band(inside)}.")
        elif ratio < 0.8:
            verdict = (f"you over-quote this one (runs ×{ratio:.2f} of the "
                       f"quote) — {_fmt_band(outside)} is the honest anchor.")
        else:
            verdict = (f"well-calibrated quotes here (×{ratio:.2f}); "
                       f"{_fmt_band(inside)} stands.")
        return {"ok": True, "task_type": ttype, "inside": inside,
                "outside_anchor": round(outside, 2),
                "uplift_ratio": round(ratio, 2),
                "reference_class_size": n,
                "empirical_hit_rate": round(hit_rate, 2),
                "verdict": verdict}
    except Exception:  # noqa: BLE001 - never raises
        return {"ok": False, "reason": "check failed"}


def calibration(db_path: str = "") -> list[dict[str, Any]]:
    """Quoted bands vs reality, per task type (Tetlock-style calibration).

    Our bands are quoted as ~80% capture bands; this compares the observed
    in-band hit rate against that target. Never raises.
    """
    try:
        store = get_store(db_path)
        if store._db is None:
            return []
        rows = store._db.execute(
            "SELECT task_type, n, misses FROM estimate_stats "
            "WHERE n >= 3 ORDER BY n DESC").fetchall()
        out = []
        for r in rows:
            n, misses = int(r["n"] or 0), int(r["misses"] or 0)
            hit = (n - misses) / n
            if hit < 0.60:
                verdict = "overconfident — bands miss too often; quote wider"
            elif hit > 0.95:
                verdict = "underconfident — bands wider than needed"
            else:
                verdict = "calibrated"
            out.append({"task_type": r["task_type"], "runs": n,
                        "hit_rate": round(hit, 2),
                        "target": 0.80, "verdict": verdict})
        return out
    except Exception:  # noqa: BLE001 - never raises
        return []


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
            "       /eta sim <project> <label:o/m/p> [label:o/m/p ...] — Monte Carlo\n"
            "       /eta outside <task-type> <quoted-min> — reference-class check\n"
            "       /eta calibrate — quoted bands vs reality\n"
            "example: /eta research prep=10 read=20 write=15\n"
            "example: /eta sim launch \"api:20/40/90\" \"ui:10/20/45\"")


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
        if parts[0] == "outside" and len(parts) >= 3:
            try:
                quoted = float(parts[2])
            except (TypeError, ValueError):
                return "usage: /eta outside <task-type> <quoted-min>"
            r = reference_class_check(parts[1], quoted,
                                      db_path if db_path else "")
            if not r.get("ok"):
                return "couldn't run the reference-class check."
            lines = [f"🏛️ outside view for {r['task_type']}:"]
            lines.append(f"   inside view (your quote): {_fmt_band(r['inside'])}")
            if r.get("reference_class_size", 0) >= 3:
                lines.append(f"   outside anchor: {_fmt_band(r['outside_anchor'])} "
                             f"(uplift ×{r['uplift_ratio']}, "
                             f"{r['reference_class_size']} past runs, "
                             f"hit rate {r['empirical_hit_rate']:.0%})")
            lines.append(f"   → {r['verdict']}")
            lines.append(ESTIMATE_DISCLAIMER)
            return "\n".join(lines)
        if parts[0] == "calibrate":
            rows = calibration(db_path if db_path else "")
            if not rows:
                return ("📏 no task types with enough history yet "
                        "(need 3+ recorded runs). " + ESTIMATE_DISCLAIMER)
            lines = ["📏 calibration — quoted ~80% bands vs reality:"]
            for r in rows[:12]:
                flag = "✅" if r["verdict"] == "calibrated" else "⚠️"
                lines.append(f"   {flag} {r['task_type']}: hit "
                             f"{r['hit_rate']:.0%} over {r['runs']} runs — "
                             f"{r['verdict']}")
            lines.append(ESTIMATE_DISCLAIMER)
            return "\n".join(lines)
        if parts[0] == "sim" and len(parts) >= 3:
            project = parts[1]
            steps: list[dict[str, Any]] = []
            # "label:o/m/p" or "label:o/m/p>dep1,dep2"
            for tok in parts[2:]:
                try:
                    if ">" in tok:
                        spec, depstr = tok.split(">", 1)
                        dep_ids = [d.strip() for d in depstr.split(",")
                                   if d.strip()]
                    else:
                        spec, dep_ids = tok, []
                    label, triple = spec.split(":", 1)
                    o, m, p = [float(x) for x in triple.split("/")]
                    steps.append({"id": label.strip() or f"step_{len(steps)}",
                                  "label": label.strip(),
                                  "optimistic": o, "most_likely": m,
                                  "pessimistic": p,
                                  "depends_on": dep_ids})
                except (ValueError, IndexError):
                    continue
            if not steps:
                return ("usage: /eta sim <project> <label:o/m/p> ...\n"
                        "example: /eta sim launch \"api:20/40/90\" \"ui:10/20/45>api\"")
            res = monte_carlo(project, steps)
            return res.format() + "\n" + ESTIMATE_DISCLAIMER
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
