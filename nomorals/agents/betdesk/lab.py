"""The crash pattern lab — where the time hypothesis goes to be tested.

Owner's hypothesis (2026-10-22): crash-game patterns "have to do with
time."  This module tests that — and every neighbouring claim — against
timestamped round data with real statistics, and renders a verdict per
game: an edge, or NO-EDGE, in writing (GoldenSwing doctrine: a study
maps where something works; it never assumes an edge exists).

The battery, per (provider, game), all stdlib:

* ``distribution`` — does the bust-multiplier distribution match the
  standard crash law ``P(X >= x) = (1 - e) / x`` (KS test on the
  probability-integral transform)?  A game whose base distribution is
  off-model is a different animal from one with time structure.
* ``time_of_day`` — the owner's hypothesis directly: chi-square
  homogeneity of the success rate (crash >= target) across 24 UTC
  hour bins.
* ``gap_effect`` — inter-round timing: does the gap since the previous
  round predict the next round's success (chi-square over gap
  quartiles, plus Pearson on log-gap vs log-multiplier)?
* ``autocorr`` — Ljung-Box portmanteau on log-multipliers, lags 1..10:
  is a round's outcome serially dependent on its predecessors?
* ``streak`` — conditional success rate after success vs after
  failure (2x2 chi-square): the classic "a low one is due" claim.
* ``runs`` — Wald-Wolfowitz runs test on the success sequence.

Honesty machinery, not decoration:

* Every p-value in a game's battery is Bonferroni-corrected across
  the battery before anything is called significant.
* An EDGE_CANDIDATE additionally needs split-half confirmation: the
  structure must be discovered on the first half of the data and
  re-confirmed on the second half.  One-sample significance on
  self-selected structure is how fake edges are born.
* A verdict names the sample size behind it, and below
  ``MIN_ROUNDS`` the only honest verdict is INSUFFICIENT_DATA.
* Only a confirmed *time* structure (``time_of_day``) is callable by
  the signal engine — a timed alert needs a clock to point at.
  Other confirmed structures are recorded as real findings that are
  NOT callable, and say why.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .crash import CrashRound

__all__ = [
    "MIN_ROUNDS", "TestResult", "GameVerdict", "analyze_game",
    "chi2_sf", "normal_sf", "ks_uniform",
    "verify_hash_chain", "bustabit_crash_from_hash",
]

#: Below this many rounds, no verdict but INSUFFICIENT_DATA is honest.
MIN_ROUNDS = 300

#: Canonical cashout target the success/failure tests are framed on.
DEFAULT_TARGET = 2.0


# ── distribution machinery (stdlib) ───────────────────────────────────

def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) — series/continued
    fraction (Numerical Recipes gammp/gammq), accurate to ~1e-12."""
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # series for P(a, x); Q = 1 - P
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))
    # continued fraction for Q(a, x) directly
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def chi2_sf(stat: float, dof: int) -> float:
    """Survival function of the chi-square distribution."""
    if dof <= 0 or stat <= 0:
        return 1.0
    return _gammaincc(dof / 2.0, stat / 2.0)


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_stat(observed: Sequence[float], expected: Sequence[float]) -> float:
    return sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)


def ks_uniform(values: Sequence[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov test of samples against Uniform(0, 1).

    Returns (D, p) with the standard asymptotic p-value.
    """
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0, 1.0
    d = 0.0
    for i, x in enumerate(xs):
        d = max(d, (i + 1) / n - x, x - i / n)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    # Kolmogorov distribution survival
    p = 0.0
    for j in range(1, 101):
        p += 2 * ((-1) ** (j - 1)) * math.exp(-2 * (j * lam) ** 2)
        if j > 10 and math.exp(-2 * (j * lam) ** 2) < 1e-12:
            break
    return d, max(0.0, min(1.0, p))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


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
class GameVerdict:
    provider: str
    game: str
    n_rounds: int
    n_timed: int                       # rounds carrying a real timestamp
    target: float = DEFAULT_TARGET
    house_edge: float = 0.01
    verdict: str = "INSUFFICIENT_DATA"  # NO_EDGE | EDGE_CANDIDATE | INSUFFICIENT_DATA
    tests: list = field(default_factory=list)        # [TestResult]
    flagged: list = field(default_factory=list)      # test names surviving correction
    confirmed: list = field(default_factory=list)    # flagged AND split-half confirmed
    callable_structure: str = ""       # "time_of_day" when signals may fire
    evidence: str = ""                 # human summary of effect sizes
    what_would_change: str = ""
    base_success_rate: float = 0.0
    best_hours: list = field(default_factory=list)   # UTC hours, when tod flags
    hour_rates: dict = field(default_factory=dict)   # hour -> success rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "game": self.game,
            "n_rounds": self.n_rounds, "n_timed": self.n_timed,
            "target": self.target, "house_edge": self.house_edge,
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


# ── individual tests ───────────────────────────────────────────────────

def _successes(rounds: Sequence[CrashRound], target: float) -> list[int]:
    return [1 if r.crash >= target else 0 for r in rounds]


def _test_distribution(rounds: Sequence[CrashRound],
                       house_edge: float) -> TestResult:
    """KS test of bust multipliers against the standard crash law."""
    us = []
    for r in rounds:
        if r.crash >= 1.0:
            us.append(min(1.0, (1.0 - house_edge) / r.crash))
    if len(us) < 50:
        return TestResult("distribution", skipped="fewer than 50 usable rounds")
    d, p = ks_uniform(us)
    return TestResult(
        "distribution", statistic=d, p_value=p,
        detail=(f"KS D={d:.4f} vs crash law P(X>=x)=(1-{house_edge})/x "
                f"over {len(us)} rounds"))


def _tod_table(rounds: Sequence[CrashRound], target: float):
    """(per-hour [successes, total], base_rate) from timestamped rounds."""
    hours: dict[int, list[int]] = {h: [0, 0] for h in range(24)}
    tot = suc = 0
    for r in rounds:
        if not r.ts:
            continue
        h = int((r.ts // 3600) % 24)
        hours[h][1] += 1
        tot += 1
        if r.crash >= target:
            hours[h][0] += 1
            suc += 1
    return hours, (suc / tot if tot else 0.0)


def _chi2_homogeneity(hours: dict[int, list[int]]) -> Optional[tuple]:
    """Chi-square across hour bins with adequate expected counts.

    Returns (stat, dof, used_hours) or None when too many bins are
    starved (expected successes or failures < 5 in >20% of bins).
    """
    used = {h: v for h, v in hours.items() if v[1] > 0}
    if len(used) < 4:
        return None
    tot_s = sum(v[0] for v in used.values())
    tot_n = sum(v[1] for v in used.values())
    if tot_n == 0:
        return None
    base = tot_s / tot_n
    starved = 0
    obs, exp = [], []
    for h in sorted(used):
        s, n = used[h]
        es, ef = n * base, n * (1 - base)
        if es < 5 or ef < 5:
            starved += 1
        obs += [s, n - s]
        exp += [es, ef]
    if starved > 0.2 * len(used):
        return None
    stat = _chi2_stat(obs, exp)
    return stat, len(used) - 1, sorted(used)


def _test_time_of_day(rounds: Sequence[CrashRound],
                      target: float) -> tuple[TestResult, dict, list, float]:
    hours, base = _tod_table(rounds, target)
    rates = {h: (v[0] / v[1] if v[1] else 0.0) for h, v in hours.items()}
    res = _chi2_homogeneity(hours)
    if res is None:
        return (TestResult("time_of_day",
                           skipped="timestamped rounds too sparse per hour"),
                hours, [], base)
    stat, dof, used = res
    p = chi2_sf(stat, dof)
    ranked = sorted(used, key=lambda h: rates[h], reverse=True)
    spread = (max(rates[h] for h in used) - min(rates[h] for h in used)) if used else 0.0
    return (TestResult(
        "time_of_day", statistic=stat, p_value=p,
        detail=(f"chi2={stat:.2f} dof={dof} across {len(used)} hour bins; "
                f"success-rate spread {spread:.1%} around base {base:.1%}")),
        hours, ranked, base)


def _test_gap(rounds: Sequence[CrashRound], target: float) -> TestResult:
    timed = [r for r in rounds if r.ts]
    if len(timed) < 100:
        return TestResult("gap_effect", skipped="fewer than 100 timestamped rounds")
    gaps = [timed[i + 1].ts - timed[i].ts for i in range(len(timed) - 1)]
    nxt = [1 if timed[i + 1].crash >= target else 0 for i in range(len(timed) - 1)]
    if any(g < 0 for g in gaps):
        return TestResult("gap_effect", skipped="timestamps not monotonic")
    srt = sorted(gaps)
    qs = [srt[len(srt) // 4], srt[len(srt) // 2], srt[3 * len(srt) // 4]]

    def bucket(g: float) -> int:
        return sum(1 for q in qs if g > q)

    cells: dict[int, list[int]] = {b: [0, 0] for b in range(4)}
    for g, s in zip(gaps, nxt):
        b = bucket(g)
        cells[b][1] += 1
        cells[b][0] += s
    tot_s = sum(v[0] for v in cells.values())
    tot_n = sum(v[1] for v in cells.values())
    base = tot_s / tot_n if tot_n else 0.0
    obs, exp = [], []
    for b in range(4):
        s, n = cells[b]
        obs += [s, n - s]
        exp += [n * base, n * (1 - base)]
    if min(exp) < 5:
        return TestResult("gap_effect", skipped="gap buckets too sparse")
    stat = _chi2_stat(obs, exp)
    p = chi2_sf(stat, 3)
    lg = [math.log(max(g, 1e-6)) for g in gaps]
    lc = [math.log(timed[i + 1].crash) for i in range(len(timed) - 1)]
    r = _pearson(lg, lc)
    return TestResult(
        "gap_effect", statistic=stat, p_value=p,
        detail=(f"chi2={stat:.2f} dof=3 across gap quartiles; "
                f"Pearson(log gap, log next crash)={r:+.3f}"))


def _test_autocorr(rounds: Sequence[CrashRound], max_lag: int = 10) -> TestResult:
    xs = [math.log(r.crash) for r in rounds if r.crash > 0]
    n = len(xs)
    if n < 100:
        return TestResult("autocorr", skipped="fewer than 100 rounds")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs)
    if var <= 0:
        return TestResult("autocorr", skipped="zero variance")
    q = 0.0
    for k in range(1, max_lag + 1):
        num = sum((xs[t] - mean) * (xs[t - k] - mean) for t in range(k, n))
        rk = num / var
        q += (rk * rk) / (n - k)
    q *= n * (n + 2)
    p = chi2_sf(q, max_lag)
    return TestResult(
        "autocorr", statistic=q, p_value=p,
        detail=f"Ljung-Box Q={q:.2f} dof={max_lag} on log-multipliers")


def _test_streak(rounds: Sequence[CrashRound], target: float) -> TestResult:
    s = _successes(rounds, target)
    if len(s) < 100:
        return TestResult("streak", skipped="fewer than 100 rounds")
    # 2x2: rows = previous outcome, cols = next outcome (success counts)
    a = sum(1 for i in range(len(s) - 1) if s[i] == 1 and s[i + 1] == 1)
    b = sum(1 for i in range(len(s) - 1) if s[i] == 1 and s[i + 1] == 0)
    c = sum(1 for i in range(len(s) - 1) if s[i] == 0 and s[i + 1] == 1)
    d = sum(1 for i in range(len(s) - 1) if s[i] == 0 and s[i + 1] == 0)
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return TestResult("streak", skipped="degenerate outcome sequence")
    stat = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    p = chi2_sf(stat, 1)
    p_after_win = a / (a + b) if (a + b) else 0.0
    p_after_loss = c / (c + d) if (c + d) else 0.0
    return TestResult(
        "streak", statistic=stat, p_value=p,
        detail=(f"P(success|prev success)={p_after_win:.1%} vs "
                f"P(success|prev fail)={p_after_loss:.1%}"))


def _test_runs(rounds: Sequence[CrashRound], target: float) -> TestResult:
    s = _successes(rounds, target)
    n = len(s)
    if n < 100:
        return TestResult("runs", skipped="fewer than 100 rounds")
    n1 = sum(s)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return TestResult("runs", skipped="only one outcome class")
    runs = 1 + sum(1 for i in range(1, n) if s[i] != s[i - 1])
    mu = 2 * n1 * n0 / n + 1
    var = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    if var <= 0:
        return TestResult("runs", skipped="degenerate runs variance")
    z = (runs - mu) / math.sqrt(var)
    p = 2 * normal_sf(abs(z))
    return TestResult(
        "runs", statistic=z, p_value=min(1.0, p),
        detail=f"{runs} runs vs {mu:.0f} expected (z={z:+.2f})")


# ── split-half confirmation ────────────────────────────────────────────

def _confirm_time_of_day(rounds: Sequence[CrashRound],
                         target: float) -> tuple[bool, list[int], str]:
    """Discover flagged hours on half 1; test them as a group on half 2."""
    timed = [r for r in rounds if r.ts]
    if len(timed) < 200:
        return False, [], "too few timestamped rounds to split"
    half = len(timed) // 2
    h1, h2 = timed[:half], timed[half:]
    hours1, base1 = _tod_table(h1, target)
    rates1 = {h: (v[0] / v[1] if v[1] else 0.0) for h, v in hours1.items()}
    counts1 = {h: v[1] for h, v in hours1.items()}
    eligible = [h for h in range(24) if counts1.get(h, 0) >= 20]
    if len(eligible) < 4:
        return False, [], "first half too sparse per hour"
    ranked = sorted(eligible, key=lambda h: rates1[h], reverse=True)
    flagged = ranked[:max(1, len(eligible) // 4)]
    # half 2: flagged hours vs the rest, 2x2 chi-square
    def group(rs: Sequence[CrashRound]):
        a = b = 0
        for r in rs:
            h = int((r.ts // 3600) % 24)
            if h in flagged:
                a += 1 if r.crash >= target else 0
                b += 0 if r.crash >= target else 1
        return a, b
    fa, fb = group(h2)
    ra = sum(1 for r in h2 if int((r.ts // 3600) % 24) not in flagged
             and r.crash >= target)
    rb = sum(1 for r in h2 if int((r.ts // 3600) % 24) not in flagged
             and r.crash < target)
    n = fa + fb + ra + rb
    if min(fa + fb, ra + rb, fa + ra, fb + rb) == 0 or n == 0:
        return False, flagged, "second half degenerate for flagged hours"
    stat = n * (fa * rb - fb * ra) ** 2 / ((fa + fb) * (ra + rb) *
                                           (fa + ra) * (fb + rb))
    p = chi2_sf(stat, 1)
    rate_flag = fa / (fa + fb) if (fa + fb) else 0.0
    rate_rest = ra / (ra + rb) if (ra + rb) else 0.0
    ok = p < 0.05 and rate_flag > rate_rest
    return ok, flagged, (f"half-2 flagged-hour success {rate_flag:.1%} vs "
                          f"rest {rate_rest:.1%} (chi2 p={p:.4f})")


def _confirm_simple(rounds: Sequence[CrashRound], test_name: str,
                    target: float, house_edge: float) -> tuple[bool, str]:
    """Re-run a non-time test on the second half alone."""
    half = len(rounds) // 2
    second = rounds[half:]
    fn = {"distribution": lambda: _test_distribution(second, house_edge),
          "gap_effect": lambda: _test_gap(second, target),
          "autocorr": lambda: _test_autocorr(second),
          "streak": lambda: _test_streak(second, target),
          "runs": lambda: _test_runs(second, target)}.get(test_name)
    if fn is None:
        return False, "no confirmation path"
    res = fn()
    if not res.ran:
        return False, f"second half: {res.skipped}"
    return res.p_value < 0.05, f"half-2 {test_name} p={res.p_value:.4f}"


# ── the battery ────────────────────────────────────────────────────────

def analyze_game(rounds: Sequence[CrashRound], provider: str = "",
                 game: str = "", target: float = DEFAULT_TARGET,
                 house_edge: float = 0.01) -> GameVerdict:
    """Run the full battery over one game's rounds and render a verdict."""
    rounds = list(rounds)
    v = GameVerdict(provider=provider or (rounds[0].provider if rounds else ""),
                    game=game or (rounds[0].game if rounds else ""),
                    n_rounds=len(rounds),
                    n_timed=sum(1 for r in rounds if r.ts),
                    target=target, house_edge=house_edge)
    if len(rounds) < MIN_ROUNDS:
        v.evidence = (f"{len(rounds)} rounds — below the {MIN_ROUNDS}-round "
                      f"floor for an honest verdict.")
        v.what_would_change = (f"collect at least {MIN_ROUNDS} timestamped "
                               f"rounds for this game")
        return v

    tod, hours, _ranked, base = _test_time_of_day(rounds, target)
    v.hour_rates = {h: (c[0] / c[1] if c[1] else 0.0)
                    for h, c in hours.items()}
    v.base_success_rate = base
    v.tests = [
        _test_distribution(rounds, house_edge),
        tod,
        _test_gap(rounds, target),
        _test_autocorr(rounds),
        _test_streak(rounds, target),
        _test_runs(rounds, target),
    ]
    ran = [t for t in v.tests if t.ran]
    m = max(1, len(ran))
    flagged = [t.name for t in ran if t.p_value * m < 0.01]
    v.flagged = flagged

    confirmed: list[str] = []
    notes: list[str] = []
    for name in flagged:
        if name == "time_of_day":
            ok, best, note = _confirm_time_of_day(rounds, target)
            if ok:
                confirmed.append(name)
                v.best_hours = best
                v.callable_structure = "time_of_day"
            notes.append(f"time_of_day split-half: {note}")
        else:
            ok, note = _confirm_simple(rounds, name, target, house_edge)
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
            v.what_would_change = ("more rounds shrinking the flagged hours' "
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
                               "on freshly collected rounds")
    else:
        v.verdict = "NO_EDGE"
        v.evidence = (f"no test in the battery survived Bonferroni "
                      f"correction (x{m}). Battery: {sig}.")
        v.what_would_change = ("a structure strong enough to survive "
                               "correction AND split-half confirmation on "
                               "fresh rounds")
    return v


# ── provably-fair verification ─────────────────────────────────────────

def verify_hash_chain(rounds: Sequence[CrashRound]) -> dict[str, Any]:
    """Verify sha256 hash-chain linkage over rounds carrying hashes.

    Many crash providers chain rounds: ``sha256(hash_n) == hash_{n-1}``
    (each round publishes the hash of the NEXT round in advance, and
    yesterday's reveal must hash forward to today's).  Wherever two
    consecutive rounds carry (game_hash, prev_hash) material, the link
    is checked; rounds without material are counted, not assumed.
    """
    checked = ok = 0
    failures: list[str] = []
    for r in rounds:
        if not r.game_hash or not r.prev_hash:
            continue
        checked += 1
        digest = hashlib.sha256(r.game_hash.encode("utf-8")).hexdigest()
        if digest == r.prev_hash.lower():
            ok += 1
        else:
            failures.append(r.round_id)
    return {"checked": checked, "ok": ok, "failed": len(failures),
            "failure_rounds": failures[:10],
            "verdict": ("chain intact over checked links"
                        if checked and ok == checked else
                        "chain breaks found" if checked else
                        "no chain material published in these rounds")}


def bustabit_crash_from_hash(game_hash_hex: str, salt: str = "") -> float:
    """Bustabit's published provably-fair crash derivation.

    Per bustabit's public fairness disclosure: the crash point derives
    from HMAC-SHA256(key=salt, msg=game hash); the first 52 bits give
    ``h``; if ``h % 33 == 0`` the round is an instant bust (the house
    edge event), otherwise ``crash = floor(100 / (1 - h / 2^52)) / 100``
    bounded below at 1.00.  Implemented so revealed (hash, salt) pairs
    can be re-derived and compared against the published bust — a
    mismatch means the provider's disclosure and its rounds disagree.
    """
    mac = _hmac.new(salt.encode("utf-8"), game_hash_hex.encode("utf-8"),
                    hashlib.sha256).hexdigest()
    h = int(mac[:13], 16)  # 52 bits
    if h % 33 == 0:
        return 1.00
    crash = math.floor(100.0 / (1.0 - h / float(1 << 52))) / 100.0
    return max(1.00, crash)
