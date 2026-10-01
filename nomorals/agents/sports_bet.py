"""God-tier sports bet analyst — ensemble ML, value detection, Kelly staking.

Analysis only.  This module computes probabilities, expected value and stake
sizes from team ratings, form, goal models and market odds.  It never places
bets and contains no bet-placement code paths — there is nothing here that
talks to a bookmaker account.

The ensemble (all stdlib, no sklearn):
    1. EloModel        — Elo ratings + home advantage + margin-of-victory,
                         mapped to 1X2 with a draw-rate model.
    2. PoissonModel    — Dixon/Coles-lite attack/defense strengths, full
                         scoreline matrix -> 1X2 and over/under 2.5.
    3. FormModel       — exponentially-decayed last-N form differential,
                         head-to-head blended in.
    4. MarketModel     — de-vigged bookmaker consensus (the "wisdom of
                         crowds" baseline every other model must beat).
    meta               — multinomial logistic regression trained from
                         scratch (full-batch gradient descent, deterministic)
                         on the base models' probability outputs; falls back
                         to Brier-weighted stacking until enough history
                         exists.

Value = model probability vs de-vigged bookmaker implied probability.
Staking = fractional Kelly, capped.  Everything is seeded and reproducible.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

__all__ = [
    "Fixture", "OddsSnapshot", "Analysis", "BacktestResult",
    "EloSystem", "FormTracker", "PoissonModel",
    "score_to_1x2", "devig_probs", "poisson_pmf",
    "EloModel", "PoissonPredictor", "FormModel", "MarketModel",
    "LogisticMeta", "EnsembleAnalyst",
    "expected_value", "kelly_fraction",
    "BetStore", "default_store",
    "OddsFetcher", "ManualFetcher", "TheOddsApiFetcher", "FootballDataFetcher",
    "synthetic_history",
]

# ── data records ─────────────────────────────────────────────────────────────


@dataclass
class Fixture:
    """One match.  home_goals/away_goals None => upcoming (unplayed)."""
    home: str
    away: str
    league: str = "GEN"
    date: str = ""
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_rest: int = 7      # days since home team's last match
    away_rest: int = 7
    closing: Optional[tuple] = None  # (home, draw, away) closing odds for CLV

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def outcome(self) -> Optional[int]:
        """0 = home win, 1 = draw, 2 = away win."""
        if not self.played:
            return None
        if self.home_goals > self.away_goals:
            return 0
        if self.home_goals == self.away_goals:
            return 1
        return 2


@dataclass
class OddsSnapshot:
    bookmaker: str
    home: float
    draw: float
    away: float
    ts: str = ""

    def as_tuple(self) -> tuple:
        return (self.home, self.draw, self.away)


@dataclass
class Analysis:
    home: str
    away: str
    league: str
    model_probs: dict          # model name -> (pH, pD, pA)
    ensemble: tuple            # (pH, pD, pA)
    fair_odds: tuple           # 1/p for ensemble
    market: Optional[tuple]    # de-vigged market (pH, pD, pA)
    best_odds: Optional[tuple] # best available (h, d, a)
    edges: list                # [(selection, model_p, odds, ev), ...] sorted
    kelly: dict                # selection -> fraction of bankroll
    briers: dict               # model name -> rolling Brier (lower better)
    meta_trained: bool


@dataclass
class BacktestResult:
    n_fixtures: int
    n_bets: int
    wins: int
    staked: float
    returned: float
    profit: float
    roi: float
    hit_rate: float
    brier: dict
    max_drawdown: float
    clv: Optional[float]       # avg closing-line value where available
    final_bankroll: float


# ── math helpers ─────────────────────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def devig_probs(home: float, draw: float, away: float) -> tuple:
    """Strip the bookmaker margin: implied probs normalized to sum to 1."""
    inv = [1.0 / max(home, 1.01), 1.0 / max(draw, 1.01), 1.0 / max(away, 1.01)]
    s = sum(inv)
    return (inv[0] / s, inv[1] / s, inv[2] / s)


def score_to_1x2(exp_score: float, draw_rate: float = 0.25) -> tuple:
    """Map an Elo-style expected score (0..1, draws count 0.5) to 1X2.

    Draws are likelier when the matchup is even; the draw mass is split
    evenly off both sides of the expected score.
    """
    exp_score = min(max(exp_score, 0.01), 0.99)
    evenness = 1.0 - abs(2.0 * exp_score - 1.0)
    d = draw_rate * (0.6 + 0.8 * evenness)
    d = min(max(d, 0.05), 0.45)
    p_h = exp_score - d / 2.0
    p_a = 1.0 - exp_score - d / 2.0
    p_h = max(p_h, 0.01)
    p_a = max(p_a, 0.01)
    s = p_h + d + p_a
    return (p_h / s, d / s, p_a / s)


def expected_value(p_model: float, odds: float) -> float:
    """EV of a 1-unit stake: p*odds - 1.  Positive => value."""
    return p_model * odds - 1.0


def kelly_fraction(p: float, odds: float, frac: float = 0.5,
                   cap: float = 0.05) -> float:
    """Fractional Kelly stake as a fraction of bankroll, capped.

    f* = (b*p - q) / b,  b = decimal odds - 1.  Negative => no bet (0).
    """
    b = odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    if f <= 0:
        return 0.0
    return min(f * frac, cap)


def brier_score(probs: tuple, outcome: int) -> float:
    """Multi-class Brier: mean squared error vs one-hot outcome."""
    return sum((probs[i] - (1.0 if i == outcome else 0.0)) ** 2
               for i in range(3)) / 3.0


# ── Elo ──────────────────────────────────────────────────────────────────────

class EloSystem:
    """Elo with home advantage and margin-of-victory multiplier (538-style)."""

    def __init__(self, base: float = 1500.0, k: float = 20.0,
                 hfa: float = 65.0):
        self.base = base
        self.k = k
        self.hfa = hfa
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected(self, home: str, away: str) -> float:
        diff = self.rating(home) - self.rating(away) + self.hfa
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, home: str, away: str, home_goals: int,
               away_goals: int) -> None:
        rh, ra = self.rating(home), self.rating(away)
        exp = 1.0 / (1.0 + 10.0 ** (-(rh - ra + self.hfa) / 400.0))
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0
        margin = abs(home_goals - away_goals)
        # 538 NFL-style MOV multiplier, softened for soccer scorelines
        mov_mult = ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * abs(rh - ra))
        mov_mult = min(max(mov_mult, 0.5), 2.0)
        shift = self.k * mov_mult * (actual - exp)
        self.ratings[home] = rh + shift
        self.ratings[away] = ra - shift

    def to_dict(self) -> dict:
        return {"base": self.base, "k": self.k, "hfa": self.hfa,
                "ratings": dict(self.ratings)}

    @classmethod
    def from_dict(cls, d: dict) -> "EloSystem":
        e = cls(base=d.get("base", 1500.0), k=d.get("k", 20.0),
                hfa=d.get("hfa", 65.0))
        e.ratings = dict(d.get("ratings", {}))
        return e


# ── form ─────────────────────────────────────────────────────────────────────

class FormTracker:
    """Exponentially-decayed last-N form (points per game, 0..3)."""

    def __init__(self, n: int = 5, decay: float = 0.8):
        self.n = n
        self.decay = decay
        self._results: dict[str, list] = {}  # team -> [(seq, points)]
        self._seq = 0

    def add(self, team: str, points: float) -> None:
        self._seq += 1
        self._results.setdefault(team, []).append((self._seq, points))

    def form(self, team: str) -> float:
        rows = self._results.get(team, [])
        if not rows:
            return 1.35  # league-average-ish prior
        rows = rows[-self.n:]
        num = den = 0.0
        for i, (_, pts) in enumerate(reversed(rows)):
            w = self.decay ** i
            num += w * pts
            den += w
        return num / den if den else 1.35

    def diff(self, home: str, away: str) -> float:
        """Form differential in points/game, home-adjusted."""
        return (self.form(home) - self.form(away)) + 0.15


# ── Poisson goal model ───────────────────────────────────────────────────────

class PoissonModel:
    """Dixon/Coles-lite: team attack/defense strengths vs league averages."""

    def __init__(self):
        self.home_avg = 1.45
        self.away_avg = 1.15
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}

    def fit(self, fixtures: list) -> None:
        scored_h: dict[str, float] = {}
        conc_h: dict[str, float] = {}
        n_h: dict[str, int] = {}
        scored_a: dict[str, float] = {}
        conc_a: dict[str, float] = {}
        n_a: dict[str, int] = {}
        th = ta = 0.0
        n = 0
        for f in fixtures:
            if not f.played:
                continue
            n += 1
            th += f.home_goals
            ta += f.away_goals
            scored_h[f.home] = scored_h.get(f.home, 0.0) + f.home_goals
            conc_h[f.home] = conc_h.get(f.home, 0.0) + f.away_goals
            n_h[f.home] = n_h.get(f.home, 0) + 1
            scored_a[f.away] = scored_a.get(f.away, 0.0) + f.away_goals
            conc_a[f.away] = conc_a.get(f.away, 0.0) + f.home_goals
            n_a[f.away] = n_a.get(f.away, 0) + 1
        if n == 0:
            return
        self.home_avg = th / n
        self.away_avg = ta / n
        teams = set(scored_h) | set(scored_a)
        for t in teams:
            hs = scored_h.get(t, 0.0) / max(n_h.get(t, 1), 1)
            hc = conc_h.get(t, 0.0) / max(n_h.get(t, 1), 1)
            aws = scored_a.get(t, 0.0) / max(n_a.get(t, 1), 1)
            awc = conc_a.get(t, 0.0) / max(n_a.get(t, 1), 1)
            # blend home/away splits toward overall, shrink small samples
            played = n_h.get(t, 0) + n_a.get(t, 0)
            shrink = played / (played + 6.0)
            att = (((hs + aws) / 2.0) / ((self.home_avg + self.away_avg) / 2.0)
                   if (self.home_avg + self.away_avg) else 1.0)
            dfn = (((hc + awc) / 2.0) / ((self.home_avg + self.away_avg) / 2.0)
                   if (self.home_avg + self.away_avg) else 1.0)
            self.attack[t] = 1.0 + (att - 1.0) * shrink
            self.defense[t] = 1.0 + (dfn - 1.0) * shrink

    def lambdas(self, home: str, away: str, home_rest: int = 7,
                away_rest: int = 7) -> tuple:
        lam_h = (self.home_avg * self.attack.get(home, 1.0)
                 * self.defense.get(away, 1.0))
        lam_a = (self.away_avg * self.attack.get(away, 1.0)
                 * self.defense.get(home, 1.0))
        # congestion: <=2 days rest shaves a little off expected goals
        if home_rest <= 2:
            lam_h *= 0.96
        if away_rest <= 2:
            lam_a *= 0.96
        return (max(lam_h, 0.05), max(lam_a, 0.05))

    def matrix(self, home: str, away: str, home_rest: int = 7,
               away_rest: int = 7, max_goals: int = 8) -> list:
        lam_h, lam_a = self.lambdas(home, away, home_rest, away_rest)
        return [[poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
                 for j in range(max_goals + 1)]
                for i in range(max_goals + 1)]

    def predict(self, home: str, away: str, home_rest: int = 7,
                away_rest: int = 7) -> dict:
        m = self.matrix(home, away, home_rest, away_rest)
        p_h = p_d = p_a = over = 0.0
        for i, row in enumerate(m):
            for j, p in enumerate(row):
                if i > j:
                    p_h += p
                elif i == j:
                    p_d += p
                else:
                    p_a += p
                if i + j > 2.5:
                    over += p
        s = p_h + p_d + p_a
        return {"1x2": (p_h / s, p_d / s, p_a / s),
                "over25": over / s, "under25": 1.0 - over / s}

# ── base models ──────────────────────────────────────────────────────────────

class _Base:
    name = "base"

    def probs(self, home: str, away: str, ctx: dict) -> tuple:
        raise NotImplementedError


class EloModel(_Base):
    """Elo expected score -> 1X2 via the draw model, H2H blended in."""
    name = "elo"

    def probs(self, home: str, away: str, ctx: dict) -> tuple:
        elo: EloSystem = ctx["elo"]
        exp = elo.expected(home, away)
        # head-to-head: last meetings nudge the expectation a little
        h2h = ctx.get("h2h", {}).get((home, away))
        if h2h:
            hw, d, aw = h2h
            tot = hw + d + aw
            if tot:
                h2h_exp = (hw + 0.5 * d) / tot
                exp = 0.85 * exp + 0.15 * h2h_exp
        return score_to_1x2(exp, ctx.get("draw_rate", 0.25))


class PoissonPredictor(_Base):
    """Full scoreline matrix -> 1X2."""
    name = "poisson"

    def probs(self, home: str, away: str, ctx: dict) -> tuple:
        pm: PoissonModel = ctx["poisson"]
        fx: Optional[Fixture] = ctx.get("fixture")
        hr = fx.home_rest if fx else 7
        ar = fx.away_rest if fx else 7
        return pm.predict(home, away, hr, ar)["1x2"]


class FormModel(_Base):
    """Recent-form differential -> 1X2."""
    name = "form"
    PTS_TO_ELO = 50.0  # 1.0 pts/game form edge ~= 50 Elo

    def probs(self, home: str, away: str, ctx: dict) -> tuple:
        form: FormTracker = ctx["form"]
        diff = form.diff(home, away)
        exp = 1.0 / (1.0 + 10.0 ** (-(diff * self.PTS_TO_ELO) / 400.0))
        return score_to_1x2(exp, ctx.get("draw_rate", 0.25))


class MarketModel(_Base):
    """De-vigged bookmaker consensus.  The baseline to beat."""
    name = "market"

    def probs(self, home: str, away: str, ctx: dict) -> tuple:
        odds_list = ctx.get("odds") or []
        if not odds_list:
            return (0.44, 0.26, 0.30)  # flat prior when no market
        acc = [0.0, 0.0, 0.0]
        for o in odds_list:
            p = devig_probs(o.home, o.draw, o.away)
            for i in range(3):
                acc[i] += p[i]
        n = len(odds_list)
        return (acc[0] / n, acc[1] / n, acc[2] / n)


BASE_MODELS: list = [EloModel, PoissonPredictor, FormModel, MarketModel]


# ── meta-learner: multinomial logistic regression from scratch ───────────────

class LogisticMeta:
    """Softmax regression on the base models' probability outputs.

    Full-batch gradient descent, zero init, no shuffling — deterministic.
    X rows are the 4 models' (pH, pD, pA) concatenated (12 features).
    """

    def __init__(self, n_models: int = 4, lr: float = 1.0,
                 iters: int = 300, l2: float = 1e-4):
        self.n_models = n_models
        self.lr = lr
        self.iters = iters
        self.l2 = l2
        self.W = [[0.0] * 3 for _ in range(n_models * 3)]
        self.b = [0.0, 0.0, 0.0]
        self.trained_rows = 0

    @staticmethod
    def _softmax(z: list) -> list:
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def _forward(self, x: list) -> list:
        z = [self.b[c] + sum(x[j] * self.W[j][c]
                             for j in range(len(x)))
             for c in range(3)]
        return self._softmax(z)

    def fit(self, X: list, y: list) -> None:
        """X: list of 12-float rows; y: list of outcome ints 0/1/2."""
        n = len(X)
        if n == 0:
            return
        dim = len(X[0])
        for _ in range(self.iters):
            gW = [[0.0] * 3 for _ in range(dim)]
            gb = [0.0, 0.0, 0.0]
            for x, yi in zip(X, y):
                p = self._forward(x)
                for c in range(3):
                    err = p[c] - (1.0 if c == yi else 0.0)
                    gb[c] += err
                    for j in range(dim):
                        gW[j][c] += err * x[j]
            for j in range(dim):
                for c in range(3):
                    self.W[j][c] -= self.lr * (gW[j][c] / n
                                               + self.l2 * self.W[j][c])
            for c in range(3):
                self.b[c] -= self.lr * (gb[c] / n)
        self.trained_rows = n

    def predict_proba(self, x: list) -> tuple:
        p = self._forward(x)
        return (p[0], p[1], p[2])

    def to_dict(self) -> dict:
        return {"W": self.W, "b": self.b, "trained_rows": self.trained_rows,
                "n_models": self.n_models}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticMeta":
        m = cls(n_models=d.get("n_models", 4))
        m.W = d.get("W", m.W)
        m.b = d.get("b", m.b)
        m.trained_rows = d.get("trained_rows", 0)
        return m


def brier_stack(model_probs: dict, briers: dict) -> tuple:
    """Inverse-Brier weighted average of base-model probabilities."""
    names = list(model_probs.keys())
    inv = []
    for nm in names:
        b = briers.get(nm, 0.25)
        inv.append(1.0 / max(b, 1e-6))
    s = sum(inv)
    out = [0.0, 0.0, 0.0]
    for nm, w in zip(names, inv):
        p = model_probs[nm]
        for i in range(3):
            out[i] += (w / s) * p[i]
    t = sum(out)
    return (out[0] / t, out[1] / t, out[2] / t)


# ── the ensemble analyst ─────────────────────────────────────────────────────

class EnsembleAnalyst:
    """Owns the models, the meta-learner, and the rolling calibration."""

    MIN_META_ROWS = 30

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.seed = seed
        self.elo = EloSystem()
        self.form = FormTracker()
        self.poisson = PoissonModel()
        self.meta = LogisticMeta()
        self.briers: dict[str, float] = {}   # rolling mean Brier per model
        self.brier_n: dict[str, int] = {}
        self.draw_rate = 0.25
        self._meta_X: list = []
        self._meta_y: list = []

    # -- learning from results ------------------------------------------------
    def _points(self, f: Fixture, team: str) -> float:
        if f.home == team:
            hg, ag = f.home_goals, f.away_goals
        else:
            hg, ag = f.away_goals, f.home_goals
        return 3.0 if hg > ag else (1.0 if hg == ag else 0.0)

    def ingest(self, f: Fixture) -> None:
        """Feed one played fixture: updates Elo, form, draw-rate prior."""
        if not f.played:
            return
        self.elo.update(f.home, f.away, f.home_goals, f.away_goals)
        self.form.add(f.home, self._points(f, f.home))
        self.form.add(f.away, self._points(f, f.away))
        # rolling draw rate
        n = getattr(self, "_draw_n", 0)
        is_draw = 1.0 if f.outcome == 1 else 0.0
        self.draw_rate = (self.draw_rate * n + is_draw) / (n + 1)
        self._draw_n = n + 1

    def refit_poisson(self, fixtures: list) -> None:
        self.poisson.fit([f for f in fixtures if f.played])

    def _h2h(self, fixtures: list, home: str, away: str,
             n: int = 5) -> Optional[tuple]:
        hw = d = aw = 0
        seen = 0
        for f in reversed(fixtures):
            if not f.played or seen >= n:
                continue
            pair = {f.home, f.away}
            if pair != {home, away}:
                continue
            seen += 1
            if f.home == home:
                if f.home_goals > f.away_goals:
                    hw += 1
                elif f.home_goals == f.away_goals:
                    d += 1
                else:
                    aw += 1
            else:
                if f.away_goals > f.home_goals:
                    hw += 1
                elif f.away_goals == f.home_goals:
                    d += 1
                else:
                    aw += 1
        return (hw, d, aw) if seen else None

    def _ctx(self, home: str, away: str, fixtures: list,
             odds: list, fixture: Optional[Fixture] = None) -> dict:
        return {
            "elo": self.elo, "form": self.form, "poisson": self.poisson,
            "odds": odds, "draw_rate": self.draw_rate,
            "h2h": {(home, away): self._h2h(fixtures, home, away)},
            "fixture": fixture,
        }

    def model_probs(self, home: str, away: str, fixtures: list,
                    odds: list,
                    fixture: Optional[Fixture] = None) -> dict:
        ctx = self._ctx(home, away, fixtures, odds, fixture)
        return {m.name: m().probs(home, away, ctx) for m in BASE_MODELS}

    def ensemble_probs(self, model_probs: dict) -> tuple:
        """Meta-learner when trained, else Brier-weighted stacking."""
        if self.meta.trained_rows >= self.MIN_META_ROWS:
            x: list = []
            for m in BASE_MODELS:
                x.extend(model_probs[m.name])
            return self.meta.predict_proba(x)
        return brier_stack(model_probs, self.briers)

    def train_meta(self, rows: list) -> None:
        """rows: [(model_probs_dict, outcome_int)]."""
        X, y = [], []
        for mp, outcome in rows:
            x: list = []
            for m in BASE_MODELS:
                x.extend(mp[m.name])
            X.append(x)
            y.append(outcome)
        if len(X) >= self.MIN_META_ROWS:
            self.meta = LogisticMeta()
            self.meta.fit(X, y)
        self._meta_X, self._meta_y = X, y

    def record_briers(self, model_probs: dict, outcome: int) -> None:
        for nm, p in model_probs.items():
            b = brier_score(p, outcome)
            n = self.brier_n.get(nm, 0)
            self.briers[nm] = (self.briers.get(nm, b) * n + b) / (n + 1)
            self.brier_n[nm] = n + 1
        ens = self.ensemble_probs(model_probs)
        b = brier_score(ens, outcome)
        n = self.brier_n.get("ensemble", 0)
        self.briers["ensemble"] = (self.briers.get("ensemble", b) * n + b) / (n + 1)
        self.brier_n["ensemble"] = n + 1

    # -- analysis --------------------------------------------------------------
    def analyze(self, home: str, away: str, league: str = "GEN",
                odds: Optional[list] = None,
                fixtures: Optional[list] = None,
                fixture: Optional[Fixture] = None,
                min_edge: float = 0.04, kelly_frac: float = 0.5,
                kelly_cap: float = 0.05) -> Analysis:
        odds = odds or []
        fixtures = fixtures or []
        mp = self.model_probs(home, away, fixtures, odds, fixture)
        ens = self.ensemble_probs(mp)
        fair = tuple(1.0 / max(p, 1e-6) for p in ens)
        market = None
        best = None
        if odds:
            dev = [devig_probs(o.home, o.draw, o.away) for o in odds]
            market = tuple(sum(d[i] for d in dev) / len(dev) for i in range(3))
            best = (max(o.home for o in odds), max(o.draw for o in odds),
                    max(o.away for o in odds))
        names = ["home", "draw", "away"]
        edges = []
        kelly = {}
        if best:
            for i, sel in enumerate(names):
                ev = expected_value(ens[i], best[i])
                if ev >= min_edge:
                    edges.append((sel, ens[i], best[i], ev))
                    kelly[sel] = kelly_fraction(ens[i], best[i],
                                                kelly_frac, kelly_cap)
        edges.sort(key=lambda e: e[3], reverse=True)
        return Analysis(home=home, away=away, league=league, model_probs=mp,
                        ensemble=ens, fair_odds=fair, market=market,
                        best_odds=best, edges=edges, kelly=kelly,
                        briers=dict(self.briers),
                        meta_trained=self.meta.trained_rows >= self.MIN_META_ROWS)

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"seed": self.seed, "elo": self.elo.to_dict(),
                "draw_rate": self.draw_rate,
                "briers": self.briers, "brier_n": self.brier_n,
                "meta": self.meta.to_dict(),
                "meta_X": self._meta_X[-500:], "meta_y": self._meta_y[-500:]}

    @classmethod
    def from_dict(cls, d: dict) -> "EnsembleAnalyst":
        a = cls(seed=d.get("seed", 7))
        a.elo = EloSystem.from_dict(d.get("elo", {}))
        a.draw_rate = d.get("draw_rate", 0.25)
        a.briers = d.get("briers", {})
        a.brier_n = d.get("brier_n", {})
        a.meta = LogisticMeta.from_dict(d.get("meta", {}))
        a._meta_X = d.get("meta_X", [])
        a._meta_y = d.get("meta_y", [])
        return a

# ── persistence ──────────────────────────────────────────────────────────────

def _bet_dir() -> str:
    override = os.environ.get("NOMORALS_BET_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "nomorals", "bet")


class BetStore:
    """JSON persistence: analyst state, bankroll, fixture history."""

    MAX_HISTORY = 2000

    def __init__(self, bet_dir: str = ""):
        self.dir = bet_dir or _bet_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "analyst.json")
        self.analyst = EnsembleAnalyst()
        self.bankroll = 1000.0
        self.history: list = []  # list of dicts: fixture + odds + outcome
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        self.analyst = EnsembleAnalyst.from_dict(d.get("analyst", {}))
        self.bankroll = float(d.get("bankroll", 1000.0))
        self.history = d.get("history", [])[-self.MAX_HISTORY:]
        # rebuild form/poisson state from history (elo persisted directly)
        for h in self.history:
            f = Fixture(**{k: v for k, v in h["fixture"].items()
                           if k in Fixture.__dataclass_fields__})
            if f.played:
                self.analyst.form.add(f.home, self.analyst._points(f, f.home))
                self.analyst.form.add(f.away, self.analyst._points(f, f.away))
        self.analyst.refit_poisson(
            [Fixture(**{k: v for k, v in h["fixture"].items()
                        if k in Fixture.__dataclass_fields__})
             for h in self.history])

    def save(self) -> None:
        d = {"analyst": self.analyst.to_dict(), "bankroll": self.bankroll,
             "history": self.history[-self.MAX_HISTORY:],
             "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        os.replace(tmp, self.path)

    def fixtures(self) -> list:
        return [Fixture(**{k: v for k, v in h["fixture"].items()
                            if k in Fixture.__dataclass_fields__})
                for h in self.history]

    def record(self, fixture: Fixture, odds: Optional[list] = None) -> None:
        self.history.append({
            "fixture": asdict(fixture),
            "odds": [asdict(o) for o in (odds or [])],
        })
        self.history = self.history[-self.MAX_HISTORY:]
        if fixture.played:
            self.analyst.ingest(fixture)
        self.save()


def default_store(bet_dir: str = "") -> BetStore:
    return BetStore(bet_dir=bet_dir)


# ── odds fetchers (pluggable; manual works with no keys) ─────────────────────

class OddsFetcher:
    """Fetch upcoming fixtures + odds.  Subclass per provider."""
    name = "base"

    def fetch(self, league: str = "", limit: int = 20) -> list:
        """Return [(Fixture, [OddsSnapshot])]."""
        raise NotImplementedError

    def available(self) -> tuple:
        return (True, "ok")


class ManualFetcher(OddsFetcher):
    """Hand-entered fixtures/odds — works day one, no keys."""
    name = "manual"

    def __init__(self, entries: Optional[list] = None):
        self.entries = entries or []

    def add(self, fixture: Fixture, odds: Optional[list] = None) -> None:
        self.entries.append((fixture, odds or []))

    def fetch(self, league: str = "", limit: int = 20) -> list:
        out = [(f, o) for f, o in self.entries
               if not league or f.league == league]
        return out[:limit]


def _http_json(url: str, headers: Optional[dict] = None,
               timeout: int = 15) -> Any:
    req = urllib.request.Request(url, headers=headers or
                                 {"User-Agent": "nomorals-bet/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class TheOddsApiFetcher(OddsFetcher):
    """the-odds-api.com — needs ODDS_API_KEY env (free tier exists)."""
    name = "the-odds-api"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "").strip()

    def available(self) -> tuple:
        return ((bool(self.api_key), "ODDS_API_KEY set")
                if self.api_key else (False, "set ODDS_API_KEY env var"))

    def fetch(self, league: str = "soccer_epl", limit: int = 20) -> list:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"the-odds-api unavailable: {why}")
        url = (f"https://api.the-odds-api.com/v4/sports/{league}/odds/"
               f"?apiKey={self.api_key}&regions=eu,uk&markets=h2h&oddsFormat=decimal")
        data = _http_json(url)
        out = []
        for ev in data[:limit]:
            f = Fixture(home=ev.get("home_team", "?"),
                        away=ev.get("away_team", "?"), league=league,
                        date=ev.get("commence_time", ""))
            snaps = []
            for bm in ev.get("bookmakers", []):
                for mk in bm.get("markets", []):
                    if mk.get("key") != "h2h":
                        continue
                prices = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
                try:
                    snaps.append(OddsSnapshot(
                        bookmaker=bm.get("title", bm.get("key", "?")),
                        home=float(prices[f.home]), draw=float(prices["Draw"]),
                        away=float(prices[f.away])))
                except (KeyError, ValueError, TypeError):
                    continue
            out.append((f, snaps))
        return out


class FootballDataFetcher(OddsFetcher):
    """football-data.org — fixtures/results, needs FOOTBALL_DATA_API_KEY."""
    name = "football-data"

    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
                        .strip())

    def available(self) -> tuple:
        return ((bool(self.api_key), "FOOTBALL_DATA_API_KEY set")
                if self.api_key else
                (False, "set FOOTBALL_DATA_API_KEY env var"))

    def fetch(self, league: str = "PL", limit: int = 20) -> list:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"football-data unavailable: {why}")
        data = _http_json(
            f"https://api.football-data.org/v4/competitions/{league}/matches"
            f"?status=SCHEDULED&limit={limit}",
            headers={"X-Auth-Token": self.api_key})
        out = []
        for m in data.get("matches", [])[:limit]:
            f = Fixture(home=m["homeTeam"]["name"], away=m["awayTeam"]["name"],
                        league=league, date=m.get("utcDate", ""))
            out.append((f, []))
        return out


# ── synthetic history (tests, demos, cold-start training) ───────────────────

def synthetic_history(n: int = 400, seed: int = 7, teams: int = 12,
                      league: str = "SYN", margin: float = 0.05,
                      noise: float = 0.06) -> list:
    """Generate fixtures with results + noisy bookmaker odds.

    Each team has a true latent strength; goals come from true Poisson
    lambdas; bookmaker odds come from the TRUE 1X2 probs plus margin and
    noise — so a good model can find genuine (small) edges.
    """
    rng = random.Random(seed)
    names = [f"Team-{i:02d}" for i in range(teams)]
    strength = {t: rng.gauss(0, 150) for t in names}
    out = []
    for i in range(n):
        home, away = rng.sample(names, 2)
        lam_h = 1.45 * math.exp((strength[home] - strength[away] + 65)
                                / 900.0)
        lam_a = 1.15 * math.exp((strength[away] - strength[home]) / 900.0)
        # Knuth Poisson
        def _pois(lam):
            L = math.exp(-lam)
            k, p = 0, 1.0
            while True:
                k += 1
                p *= rng.random()
                if p <= L:
                    return k - 1
        hg, ag = _pois(lam_h), _pois(lam_a)
        # true 1x2 from a quick matrix
        ph = pd = pa = 0.0
        for gi in range(9):
            for gj in range(9):
                p = poisson_pmf(gi, lam_h) * poisson_pmf(gj, lam_a)
                if gi > gj:
                    ph += p
                elif gi == gj:
                    pd += p
                else:
                    pa += p
        s = ph + pd + pa
        true = [ph / s, pd / s, pa / s]
        # bookmaker: true probs, margin, noise
        bm = []
        for b in range(2):
            noisy = [max(t * (1 + rng.gauss(0, noise)), 0.02) for t in true]
            tot = sum(noisy)
            # add margin then invert
            probs = [x / tot * (1 - margin) for x in noisy]
            bm.append(OddsSnapshot(
                bookmaker=f"syn-bm-{b}", home=1 / probs[0],
                draw=1 / probs[1], away=1 / probs[2]))
        f = Fixture(home=home, away=away, league=league,
                    date=f"2026-01-{(i % 28) + 1:02d}",
                    home_goals=hg, away_goals=ag)
        out.append((f, bm))
    return out


# ── backtester ───────────────────────────────────────────────────────────────

def backtest(entries: list, bankroll: float = 1000.0, seed: int = 7,
             min_edge: float = 0.04, kelly_frac: float = 0.5,
             kelly_cap: float = 0.05, meta_every: int = 50) -> BacktestResult:
    """Walk-forward backtest over [(Fixture played, [OddsSnapshot])].

    Models only ever see the past: Elo/form/poisson train on prior
    fixtures, the meta-learner refits every `meta_every` fixtures.
    """
    rng = random.Random(seed)
    analyst = EnsembleAnalyst(seed=seed)
    seen: list = []
    meta_rows: list = []
    bank = bankroll
    start = bankroll
    peak = bankroll
    max_dd = 0.0
    n_bets = wins = 0
    staked = returned = 0.0
    clv_sum = clv_n = 0
    per_model_brier: dict[str, list] = {}

    for idx, (f, odds) in enumerate(entries):
        if not f.played or not odds:
            if f.played:
                analyst.ingest(f)
                seen.append(f)
            continue
        if idx and idx % meta_every == 0 and meta_rows:
            analyst.train_meta(meta_rows)
        if idx % 25 == 0:
            analyst.refit_poisson(seen)
        mp = analyst.model_probs(f.home, f.away, seen, odds, f)
        ens = analyst.ensemble_probs(mp)
        outcome = f.outcome
        analyst.record_briers(mp, outcome)
        for nm, p in mp.items():
            per_model_brier.setdefault(nm, []).append(brier_score(p, outcome))
        per_model_brier.setdefault("ensemble", []).append(
            brier_score(ens, outcome))
        meta_rows.append((mp, outcome))

        best = (max(o.home for o in odds), max(o.draw for o in odds),
                max(o.away for o in odds))
        for i, sel in enumerate(("home", "draw", "away")):
            ev = expected_value(ens[i], best[i])
            if ev < min_edge:
                continue
            stake = bank * kelly_fraction(ens[i], best[i], kelly_frac,
                                          kelly_cap)
            if stake < 0.01:
                continue
            n_bets += 1
            staked += stake
            bank -= stake
            if outcome == i:
                wins += 1
                bank += stake * best[i]
                returned += stake * best[i]
            # closing-line value
            if f.closing:
                clv_sum += (best[i] / max(f.closing[i], 1.01) - 1.0)
                clv_n += 1
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak if peak else 0.0)
        analyst.ingest(f)
        seen.append(f)

    brier = {k: sum(v) / len(v) for k, v in per_model_brier.items() if v}
    profit = bank - start
    return BacktestResult(
        n_fixtures=len(entries), n_bets=n_bets, wins=wins,
        staked=staked, returned=returned, profit=profit,
        roi=(profit / staked) if staked else 0.0,
        hit_rate=(wins / n_bets) if n_bets else 0.0,
        brier=brier, max_drawdown=max_dd,
        clv=(clv_sum / clv_n) if clv_n else None,
        final_bankroll=bank)


# ── rendering ────────────────────────────────────────────────────────────────

def render_analysis(a: Analysis) -> str:
    L = [f"{a.home} vs {a.away}  ({a.league})"]
    L.append("model probabilities (H / D / A):")
    for nm, p in a.model_probs.items():
        L.append(f"  {nm:8s} {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}")
    e = a.ensemble
    tag = "meta-learner" if a.meta_trained else "brier-stack"
    L.append(f"ensemble [{tag}]  {e[0]:.3f} / {e[1]:.3f} / {e[2]:.3f}")
    L.append(f"fair odds:  {a.fair_odds[0]:.2f} / {a.fair_odds[1]:.2f} / "
             f"{a.fair_odds[2]:.2f}")
    if a.market:
        m = a.market
        L.append(f"market:     {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f} (de-vigged)")
    if a.best_odds:
        L.append(f"best odds:  {a.best_odds[0]:.2f} / {a.best_odds[1]:.2f} / "
                 f"{a.best_odds[2]:.2f}")
    if a.edges:
        L.append("value:")
        for sel, p, odds, ev in a.edges:
            k = a.kelly.get(sel, 0.0)
            L.append(f"  {sel:5s} model {p:.3f} @ {odds:.2f}  "
                     f"EV {ev:+.1%}  kelly {k:.2%} of bankroll")
    else:
        L.append("value: none at the current edge threshold.")
    if a.briers:
        cal = ", ".join(f"{k} {v:.3f}"
                        for k, v in sorted(a.briers.items(),
                                           key=lambda kv: kv[1]))
        L.append(f"calibration (Brier, lower better): {cal}")
    return "\n".join(L)


def render_backtest(r: BacktestResult) -> str:
    L = [f"backtest: {r.n_fixtures} fixtures, {r.n_bets} bets, "
         f"{r.wins} wins ({r.hit_rate:.1%} hit rate)"]
    L.append(f"staked {r.staked:.2f}  returned {r.returned:.2f}  "
             f"profit {r.profit:+.2f}  ROI {r.roi:+.1%}")
    L.append(f"max drawdown {r.max_drawdown:.1%}  "
             f"bankroll {r.final_bankroll:.2f}")
    if r.clv is not None:
        L.append(f"avg closing-line value {r.clv:+.2%}")
    if r.brier:
        cal = ", ".join(f"{k} {v:.4f}"
                        for k, v in sorted(r.brier.items(),
                                           key=lambda kv: kv[1]))
        L.append(f"Brier (lower better): {cal}")
    return "\n".join(L)
