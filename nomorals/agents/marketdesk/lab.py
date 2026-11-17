"""The markets pattern lab — where a symbol's structure goes to be tested.

The desk studies price bars the way the betting desk studies crash
rounds: a battery of real statistical tests over the collected
series, with a verdict per symbol in writing (GoldenSwing doctrine:
a study maps where something works; it never assumes an edge
exists).

The battery, per symbol, all stdlib.  Daily bars carry structure;
hourly bars carry the time structure:

* ``trend`` — OLS slope of log closes on time: is there a drift
  worth naming (t-test on the slope)?
* ``momentum`` — the last 20-day return against the history of
  20-day returns (z-score): is the recent move extreme?
* ``vol_regime`` — current 20-day realized volatility ranked
  against its own history: is the market unusually calm or wild?
* ``level`` — price against its 50/200-day moving averages
  (z-score of the gap in units of 50-day dispersion): extended or
  washed out?  Needs 200 bars — fewer, and the test says so.
* ``mean_reversion`` — price against its 20-day average (z-score):
  stretched from its own recent mean?
* ``autocorr`` — Ljung-Box portmanteau on log returns, lags 1..10:
  is today's move serially dependent on its predecessors?
* ``streak`` — conditional up-day rate after an up day vs after a
  down day (2x2 chi-square): the classic "up days cluster" claim.
* ``hour_effect`` — chi-square homogeneity of the up-hour rate
  across 24 UTC hour bins on hourly bars: does the clock matter?

Honesty machinery, not decoration:

* Every p-value in a symbol's battery is Bonferroni-corrected
  across the battery before anything is called significant.
* An EDGE_CANDIDATE additionally needs split-half confirmation:
  the structure must be discovered on the first half of the data
  and re-confirmed on the second half.  One-sample significance on
  self-selected structure is how fake edges are born.
* A verdict names the sample size behind it, and below
  ``MIN_BARS`` daily bars the only honest verdict is
  INSUFFICIENT_DATA.
* Only a confirmed *time* structure (``hour_effect``) is callable
  by the signal engine — a timed alert needs a clock to point at.
  Other confirmed structures are recorded as real findings that
  are NOT callable, and say why.

The distribution functions (chi-square / normal survival,
Kolmogorov-Smirnov) are the betting desk lab's — one implementation
in the codebase, reused, not re-invented.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..betdesk.lab import chi2_sf, normal_sf
from .store import Candle

__all__ = [
    "MIN_BARS", "MIN_HOURLY_BARS", "TestResult", "SymbolVerdict",
    "analyze_symbol",
]

#: Below this many daily bars, no verdict but INSUFFICIENT_DATA.
MIN_BARS = 60

#: Below this many hourly bars, the hour-effect test cannot run.
MIN_HOURLY_BARS = 300


# ── results ────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    statistic: float = 0.0
    p_value: float = 1.0
    detail: str = ""
    skipped: str = ""        # non-empty = the test honestly could not run

    @property
    def ran(self) -> bool:
        return not self.skipped


@dataclass
class SymbolVerdict:
    symbol: str
    n_daily: int
    n_hourly: int
    verdict: str = "INSUFFICIENT_DATA"  # NO_EDGE | EDGE_CANDIDATE | INSUFFICIENT_DATA
    tests: list = field(default_factory=list)        # [TestResult]
    flagged: list = field(default_factory=list)      # test names surviving correction
    confirmed: list = field(default_factory=list)    # flagged AND split-half confirmed
    callable_structure: str = ""       # "hour_effect" when signals may fire
    evidence: str = ""                 # human summary of effect sizes
    what_would_change: str = ""
    base_success_rate: float = 0.0     # up-hour rate on hourly bars
    best_hours: list = field(default_factory=list)   # UTC hours, when hour_effect flags
    hour_rates: dict = field(default_factory=dict)   # hour -> up-hour rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_daily": self.n_daily, "n_hourly": self.n_hourly,
            "verdict": self.verdict,
            "tests": [{"name": t.name, "statistic": round(t.statistic, 4),
                       "p_value": t.p_value, "detail": t.detail,
                       "skipped": t.skipped} for t in self.tests],
            "flagged": self.flagged, "confirmed": self.confirmed,
            "callable_structure": self.callable_structure,
            "evidence": self.evidence,
            "what_would_change": self.what_would_change,
            "base_success_rate": self.base_success_rate,
            "best_hours": self.best_hours,
            "hour_rates": {str(k): v for k, v in self.hour_rates.items()},
        }


# ── helpers ────────────────────────────────────────────────────────────

def _log_returns(bars: Sequence[Candle]) -> list[float]:
    closes = [b.close for b in bars if b.close > 0]
    return [math.log(closes[i + 1] / closes[i])
            for i in range(len(closes) - 1)]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact via math.comb."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return max(0.0, min(1.0, total))


# ── individual tests (daily bars) ──────────────────────────────────────

def _test_trend(bars: Sequence[Candle]) -> TestResult:
    """Mean log return vs zero (t-test on daily returns).

    The drift of a price series lives in its RETURNS, not in a
    regression of the level on time: log closes are integrated, and
    an OLS slope on an integrated series has a wildly inflated
    t-statistic (the spurious-regression shape — a driftless random
    walk "trends" at t=26).  Testing the mean return is the valid
    form of the same question.
    """
    rets = _log_returns(bars)
    if len(rets) < MIN_BARS - 1:
        return TestResult("trend", skipped="fewer than 60 daily bars")
    m = _mean(rets)
    sd = _std(rets)
    if sd <= 0:
        return TestResult("trend", skipped="zero return variance")
    se = sd / math.sqrt(len(rets))
    t = m / se
    p = 2 * normal_sf(abs(t))
    return TestResult(
        "trend", statistic=t, p_value=p,
        detail=(f"mean log return {m:+.5f}/day (t={t:+.2f}); "
                f"~{m * 100:+.2f}%/day drift"))


def _test_momentum(bars: Sequence[Candle]) -> TestResult:
    """Last 20-day return vs the history of 20-day returns (z-score)."""
    closes = [b.close for b in bars if b.close > 0]
    if len(closes) < 40:
        return TestResult("momentum", skipped="fewer than 40 daily bars")
    rets20 = [math.log(closes[i] / closes[i - 20])
              for i in range(20, len(closes))]
    last = rets20[-1]
    hist = rets20[:-1]
    sd = _std(hist)
    if sd <= 0:
        return TestResult("momentum", skipped="zero 20d-return variance")
    z = (last - _mean(hist)) / sd
    p = 2 * normal_sf(abs(z))
    return TestResult(
        "momentum", statistic=z, p_value=p,
        detail=(f"last 20d return {last:+.1%} vs history "
                f"{_mean(hist):+.1%} ± {sd:.1%} (z={z:+.2f})"))


def _test_vol_regime(bars: Sequence[Candle]) -> TestResult:
    """Current 20-day realized vol ranked against its own history."""
    rets = _log_returns(bars)
    if len(rets) < 60:
        return TestResult("vol_regime", skipped="fewer than 60 daily bars")
    vols = []
    for i in range(20, len(rets) + 1):
        window = rets[i - 20:i]
        vols.append(_std(window) * math.sqrt(252))
    if len(vols) < 2:
        return TestResult("vol_regime", skipped="too few vol windows")
    cur = vols[-1]
    hist = sorted(vols[:-1])
    rank = sum(1 for v in hist if v <= cur) / len(hist)
    p = 2 * min(rank, 1 - rank)
    return TestResult(
        "vol_regime", statistic=cur, p_value=max(p, 1e-6),
        detail=(f"20d realized vol {cur:.0%} — {rank:.0%} percentile "
                f"of its own history"))


def _test_level(bars: Sequence[Candle]) -> TestResult:
    """Price vs its 50/200-day averages (z in units of 50d dispersion).

    Needs 200 bars: the 200-day average is the level that matters,
    and a 50-bar series cannot speak about it.
    """
    closes = [b.close for b in bars if b.close > 0]
    if len(closes) < 200:
        return TestResult("level", skipped="fewer than 200 daily bars")
    ma50 = _mean(closes[-50:])
    ma200 = _mean(closes[-200:])
    sd50 = _std(closes[-50:])
    if sd50 <= 0:
        return TestResult("level", skipped="zero 50d dispersion")
    z = (closes[-1] - ma50) / sd50
    p = 2 * normal_sf(abs(z))
    state = ("above both" if closes[-1] > ma50 and closes[-1] > ma200
             else "below both" if closes[-1] < ma50 and closes[-1] < ma200
             else "mixed")
    return TestResult(
        "level", statistic=z, p_value=p,
        detail=(f"close {closes[-1]:,.2f} vs MA50 {ma50:,.2f} / MA200 "
                f"{ma200:,.2f} ({state}); z={z:+.2f}"))


def _test_mean_reversion(bars: Sequence[Candle]) -> TestResult:
    """Price vs its 20-day average (z-score): stretched from its mean?"""
    closes = [b.close for b in bars if b.close > 0]
    if len(closes) < 40:
        return TestResult("mean_reversion",
                          skipped="fewer than 40 daily bars")
    ma20 = _mean(closes[-20:])
    sd20 = _std(closes[-20:])
    if sd20 <= 0:
        return TestResult("mean_reversion",
                          skipped="zero 20d dispersion")
    z = (closes[-1] - ma20) / sd20
    p = 2 * normal_sf(abs(z))
    return TestResult(
        "mean_reversion", statistic=z, p_value=p,
        detail=(f"close {closes[-1]:,.2f} vs MA20 {ma20:,.2f} "
                f"(z={z:+.2f})"))


def _test_autocorr(bars: Sequence[Candle], max_lag: int = 10) -> TestResult:
    xs = _log_returns(bars)
    n = len(xs)
    if n < 100:
        return TestResult("autocorr", skipped="fewer than 100 daily bars")
    mean = _mean(xs)
    var = sum((x - mean) ** 2 for x in xs)
    if var <= 0:
        return TestResult("autocorr", skipped="zero variance")
    q = 0.0
    for k in range(1, max_lag + 1):
        num = sum((xs[t] - mean) * (xs[t - k] - mean)
                  for t in range(k, n))
        rk = num / var
        q += (rk * rk) / (n - k)
    q *= n * (n + 2)
    p = chi2_sf(q, max_lag)
    return TestResult(
        "autocorr", statistic=q, p_value=p,
        detail=f"Ljung-Box Q={q:.2f} dof={max_lag} on log returns")


def _test_streak(bars: Sequence[Candle]) -> TestResult:
    closes = [b.close for b in bars if b.close > 0]
    if len(closes) < 100:
        return TestResult("streak", skipped="fewer than 100 daily bars")
    s = [1 if closes[i + 1] > closes[i] else 0
         for i in range(len(closes) - 1)]
    a = sum(1 for i in range(len(s) - 1) if s[i] == 1 and s[i + 1] == 1)
    b = sum(1 for i in range(len(s) - 1) if s[i] == 1 and s[i + 1] == 0)
    c = sum(1 for i in range(len(s) - 1) if s[i] == 0 and s[i + 1] == 1)
    d = sum(1 for i in range(len(s) - 1) if s[i] == 0 and s[i + 1] == 0)
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return TestResult("streak", skipped="degenerate outcome sequence")
    stat = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    p = chi2_sf(stat, 1)
    p_after_up = a / (a + b) if (a + b) else 0.0
    p_after_down = c / (c + d) if (c + d) else 0.0
    return TestResult(
        "streak", statistic=stat, p_value=p,
        detail=(f"P(up|prev up)={p_after_up:.1%} vs "
                f"P(up|prev down)={p_after_down:.1%}"))


# ── hour effect (hourly bars) ──────────────────────────────────────────

def _hour_table(bars: Sequence[Candle]):
    """(per-hour [ups, total], base_up_rate) from hourly bars.

    An up hour is a bar whose close beats its open.
    """
    hours: dict[int, list[int]] = {h: [0, 0] for h in range(24)}
    tot = ups = 0
    for b in bars:
        if not b.ts or b.close <= 0 or b.open <= 0:
            continue
        h = int((b.ts // 3600) % 24)
        hours[h][1] += 1
        tot += 1
        if b.close > b.open:
            hours[h][0] += 1
            ups += 1
    return hours, (ups / tot if tot else 0.0)


def _test_hour_effect(bars: Sequence[Candle]) -> tuple[TestResult, dict, float]:
    """Strongest single hour vs the base up-rate (binomial, Bonferroni
    over 24 hours).

    A chi-square homogeneity test across all 24 bins dilutes a real
    effect: two hot hours among 24 quiet ones barely move the
    statistic at the bar counts a young desk holds.  The focused
    form asks the question the desk actually needs — "does ANY hour
    beat the base rate, correcting for 24 looks?" — and the
    split-half confirmation below is the gate that keeps a lucky
    hour from becoming a call.
    """
    hours, base = _hour_table(bars)
    rates = {h: (v[0] / v[1] if v[1] else 0.0) for h, v in hours.items()}
    used = {h: v for h, v in hours.items() if v[1] > 0}
    if len(bars) < MIN_HOURLY_BARS or len(used) < 4:
        return (TestResult("hour_effect",
                           skipped="fewer than 300 hourly bars"),
                hours, base)
    best_p, best_detail = 1.0, ""
    for h in sorted(used):
        s, n = used[h]
        p = _binom_sf(s, n, base)
        if p < best_p:
            best_p = p
            best_detail = (f"hour {h:02d} UTC up {s}/{n} "
                           f"({rates[h]:.0%}) vs base {base:.0%}")
    p_adj = min(1.0, best_p * 24)
    return (TestResult(
        "hour_effect", statistic=best_p, p_value=p_adj,
        detail=(f"strongest hour: {best_detail} "
                f"(binomial p={best_p:.2g}, x24 Bonferroni p={p_adj:.2g})")),
        hours, base)


def _confirm_hour_effect(bars: Sequence[Candle]) -> tuple[bool, list[int], str]:
    """Discover flagged hours on half 1; test them as a group on half 2.

    A flagged hour is one whose up-rate beats the base rate with a
    per-hour binomial p surviving Bonferroni across 24 hours at the
    0.05 family level — not just the top quartile by rate, which
    would name hours on noise.  (Discovery runs at 0.05; the
    battery-level flag that admits a test here at all is 0.01, and
    the half-2 group confirmation below is the real gate.)
    """
    timed = [b for b in bars if b.ts and b.close > 0 and b.open > 0]
    if len(timed) < 200:
        return False, [], "too few hourly bars to split"
    half = len(timed) // 2
    h1, h2 = timed[:half], timed[half:]
    hours1, base1 = _hour_table(h1)
    flagged = []
    for h in range(24):
        s, n = hours1[h]
        if n < 10:
            continue
        p = _binom_sf(s, n, base1)
        if p * 24 < 0.05 and s / n > base1:
            flagged.append(h)
    if not flagged:
        return False, [], "no hour survived per-hour Bonferroni on half 1"
    # half 2: flagged hours vs the rest, 2x2 chi-square
    fa = fb = ra = rb = 0
    for b in h2:
        h = int((b.ts // 3600) % 24)
        up = 1 if b.close > b.open else 0
        if h in flagged:
            fa += up
            fb += 1 - up
        else:
            ra += up
            rb += 1 - up
    n = fa + fb + ra + rb
    if min(fa + fb, ra + rb, fa + ra, fb + rb) == 0 or n == 0:
        return False, flagged, "second half degenerate for flagged hours"
    stat = n * (fa * rb - fb * ra) ** 2 / ((fa + fb) * (ra + rb) *
                                           (fa + ra) * (fb + rb))
    p = chi2_sf(stat, 1)
    rate_flag = fa / (fa + fb) if (fa + fb) else 0.0
    rate_rest = ra / (ra + rb) if (ra + rb) else 0.0
    ok = p < 0.05 and rate_flag > rate_rest
    return ok, flagged, (f"half-2 flagged-hour up rate {rate_flag:.1%} vs "
                          f"rest {rate_rest:.1%} (chi2 p={p:.4f})")


def _confirm_simple(daily: Sequence[Candle], hourly: Sequence[Candle],
                    test_name: str) -> tuple[bool, str]:
    """Re-run a non-time test on the second half alone."""
    bars = list(daily)
    half = len(bars) // 2
    second = bars[half:]
    fn = {"trend": lambda: _test_trend(second),
          "momentum": lambda: _test_momentum(second),
          "vol_regime": lambda: _test_vol_regime(second),
          "level": lambda: _test_level(second),
          "mean_reversion": lambda: _test_mean_reversion(second),
          "autocorr": lambda: _test_autocorr(second),
          "streak": lambda: _test_streak(second)}.get(test_name)
    if fn is None:
        return False, "no confirmation path"
    res = fn()
    if not res.ran:
        return False, f"second half: {res.skipped}"
    return res.p_value < 0.05, f"half-2 {test_name} p={res.p_value:.4f}"


# ── the battery ────────────────────────────────────────────────────────

def analyze_symbol(daily: Sequence[Candle], hourly: Sequence[Candle],
                   symbol: str = "") -> SymbolVerdict:
    """Run the full battery over one symbol's bars and render a verdict."""
    daily = sorted(daily, key=lambda b: b.ts)
    hourly = sorted(hourly, key=lambda b: b.ts)
    v = SymbolVerdict(symbol=symbol or (daily[0].symbol if daily else ""),
                      n_daily=len(daily), n_hourly=len(hourly))
    if len(daily) < MIN_BARS:
        v.evidence = (f"{len(daily)} daily bars — below the {MIN_BARS}-bar "
                      f"floor for an honest verdict.")
        v.what_would_change = (f"collect at least {MIN_BARS} daily bars "
                               f"for this symbol")
        return v

    hour_res, hours, base = _test_hour_effect(hourly)
    v.hour_rates = {h: (c[0] / c[1] if c[1] else 0.0)
                    for h, c in hours.items()}
    v.base_success_rate = base
    v.tests = [
        _test_trend(daily),
        _test_momentum(daily),
        _test_vol_regime(daily),
        _test_level(daily),
        _test_mean_reversion(daily),
        _test_autocorr(daily),
        _test_streak(daily),
        hour_res,
    ]
    ran = [t for t in v.tests if t.ran]
    m = max(1, len(ran))
    flagged = [t.name for t in ran if t.p_value * m < 0.01]
    v.flagged = flagged

    confirmed: list[str] = []
    notes: list[str] = []
    for name in flagged:
        if name == "hour_effect":
            ok, best, note = _confirm_hour_effect(hourly)
            if ok:
                confirmed.append(name)
                v.best_hours = best
                v.callable_structure = "hour_effect"
            notes.append(f"hour_effect split-half: {note}")
        else:
            ok, note = _confirm_simple(daily, hourly, name)
            if ok:
                confirmed.append(name)
            notes.append(f"{name} split-half: {note}")
    v.confirmed = confirmed

    sig = "; ".join(f"{t.name} p={t.p_value:.2e}" for t in ran)
    if confirmed:
        v.verdict = "EDGE_CANDIDATE"
        v.evidence = (f"survived Bonferroni (x{m}) AND split-half confirmation: "
                      f"{', '.join(confirmed)}. Battery: {sig}. " +
                      "; ".join(notes))
        if v.callable_structure:
            v.what_would_change = ("more bars shrinking the flagged hours' "
                                   "advantage below the effect floor, or the "
                                   "half-2 confirmation failing on fresh data")
        else:
            v.evidence += (" — confirmed structure is NOT time-based, so the "
                           "signal engine cannot call it on a clock.")
            v.what_would_change = ("a time-based structure confirming as well; "
                                   "non-time structure stays a study finding")
    elif flagged:
        v.verdict = "NO_EDGE"
        v.evidence = (f"{', '.join(flagged)} looked significant before "
                      f"split-half confirmation and did NOT confirm — the "
                      f"classic false-edge shape. Battery: {sig}. " +
                      "; ".join(notes))
        v.what_would_change = ("the same structure confirming out-of-sample "
                               "on freshly collected bars")
    else:
        v.verdict = "NO_EDGE"
        v.evidence = (f"no test in the battery survived Bonferroni "
                      f"correction (x{m}). Battery: {sig}.")
        v.what_would_change = ("a structure strong enough to survive "
                               "correction AND split-half confirmation on "
                               "fresh bars")
    return v
