"""Risk management: sizing models, stop ladders, drawdown governors.

Ported from ``sentinel/risk/manager.py`` (user's own Sentinel.py bot).
Identical math is usable in backtest and live contexts; Devon uses it for
trade plans and position sizing. Pure numpy/pandas — no exchange needed.
"""

from __future__ import annotations


from .math import clamp


# ── lazy optional deps ──────────────────────────────────────────────────
# numpy/pandas are optional. The package imports without them; functions
# that need them raise TAError with a clear install hint.

class TAError(Exception):
    """Raised when a TA operation cannot be completed."""

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise TAError("numpy is required for this operation: pip install nomorals[ta]")


def _require_pandas() -> None:
    if not _HAS_PANDAS:
        raise TAError("pandas is required for this operation: pip install nomorals[ta]")



__all__ = ["RiskManager", "position_size", "PROFILES",
           # ── sweep additions ──
           "Position", "PositionTracker", "risk_parity_weights",
           "correlation_adjusted_fraction", "chandelier_exit", "optimal_f",
           "kelly_drawdown_shrink", "portfolio_heat"]

#: Named risk profiles for the FinancialExpert (default|aggressive|conservative).
#: ``min_agreement`` is calibrated for the 4-strategy committee, where the
#: vote agreement only takes the values 0.0 (split), 0.5 (3-of-4) and 1.0
#: (unanimous) — so 0.5 means supermajority, 1.0 means unanimity.
PROFILES: dict[str, dict] = {
    "default": {
        "sizing": "fixed", "base_risk": 0.01, "stop_atr": 2.0,
        "exposure_cap": 0.80, "daily_halt": -0.03, "max_drawdown": -0.15,
        "min_agreement": 0.50, "cost_bps": 10.0,
    },
    "aggressive": {
        "sizing": "fixed", "base_risk": 0.02, "stop_atr": 1.5,
        "exposure_cap": 1.00, "daily_halt": -0.05, "max_drawdown": -0.25,
        "min_agreement": 0.50, "cost_bps": 10.0,
    },
    "conservative": {
        "sizing": "fixed", "base_risk": 0.005, "stop_atr": 2.5,
        "exposure_cap": 0.50, "daily_halt": -0.02, "max_drawdown": -0.10,
        "min_agreement": 1.00, "cost_bps": 10.0,
    },
}


class RiskManager:
    """Unified risk brain: sizing models, stop ladders, governors."""

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.sizing = str(c.get("sizing", "fixed"))
        self.base_risk = float(c.get("base_risk", 0.01))
        self.kelly_cap = float(c.get("kelly_cap", 0.25))
        self.vol_target = float(c.get("vol_target", 0.12))
        self.cppi_floor = float(c.get("cppi_floor", 0.85))
        self.cppi_mult = float(c.get("cppi_multiple", 3.0))
        self.stop_atr = float(c.get("stop_atr", 2.0))
        self.time_stop_bars = int(c.get("time_stop_bars", 120))
        self.breakeven_atr = float(c.get("breakeven_atr", 1.0))
        self.daily_halt = float(c.get("daily_halt", -0.03))
        self.max_dd = float(c.get("max_drawdown", -0.15))
        self.exposure_cap = float(c.get("exposure_cap", 0.80))
        self.peak = None
        self.day_start = None
        self.halted = False

    # ------------------------------------------------------------- sizing
    @staticmethod
    def kelly_fraction(win_rate: float, payoff: float) -> float:
        """Kelly fraction f* = (p(b+1) − 1) / b, clamped inputs."""
        p = clamp(win_rate, 0.01, 0.99)
        b = max(0.1, payoff)
        return (p * (b + 1) - 1) / b

    def size_position(self, equity: float, price: float, atr: float,
                      win_rate: float = 0.5, payoff: float = 1.5,
                      realized_vol: float | None = None,
                      periods_per_year: int = 8760) -> dict:
        """Size from risk budget: risk ``risk_frac`` of equity per stop.

        Returns ``fraction`` (of equity, capped by ``exposure_cap``),
        ``shares``, ``notional``, ``risk_frac``, ``method`` and ``stop_dist``.
        """
        price = max(1e-9, float(price))
        atr = max(1e-9, float(atr))
        risk_frac = self.base_risk
        method = self.sizing
        if method == "kelly_frac":
            k = self.kelly_fraction(win_rate, payoff)
            risk_frac = clamp(0.25 * max(0.0, k),
                              0.001, self.kelly_cap * self.base_risk * 10)
            risk_frac = clamp(risk_frac, 0.001, 0.05)
        elif method == "vol_target" and realized_vol:
            sig = max(1e-6, float(realized_vol)) * _np.sqrt(periods_per_year)
            risk_frac = clamp(self.vol_target / sig, 0.05, 2.0) \
                * self.base_risk * 4
            risk_frac = clamp(risk_frac, 0.001, 0.05)
        elif method == "cppi":
            floor = self.cppi_floor * (self.peak or equity)
            cushion = max(0.0, (equity - floor) / (equity + 1e-12))
            risk_frac = clamp(cushion * self.cppi_mult * self.base_risk * 4,
                              0.0, 0.05)
        stop_dist = self.stop_atr * atr
        risk_money = equity * risk_frac
        shares = risk_money / stop_dist
        notional = shares * price
        fraction = clamp(notional / (equity + 1e-12), 0.0, self.exposure_cap)
        shares = fraction * equity / price
        return {"fraction": float(fraction), "shares": float(shares),
                "notional": float(fraction * equity),
                "risk_frac": float(risk_frac), "method": method,
                "stop_dist": float(stop_dist)}

    # ------------------------------------------------------------- stops
    def stop_levels(self, side: int, entry: float, atr: float) -> dict:
        """ATR stop ladder: stop, breakeven trigger, 1R and 2R targets."""
        atr = max(1e-9, float(atr))
        if side >= 0:
            return {"stop": entry - self.stop_atr * atr,
                    "breakeven_trigger": entry + self.breakeven_atr * atr,
                    "target_1": entry + self.stop_atr * atr,
                    "target_2": entry + 2 * self.stop_atr * atr}
        return {"stop": entry + self.stop_atr * atr,
                "breakeven_trigger": entry - self.breakeven_atr * atr,
                "target_1": entry - self.stop_atr * atr,
                "target_2": entry - 2 * self.stop_atr * atr}

    @staticmethod
    def trailing_stop(side: int, current_stop: float, price: float,
                      atr: float, mult: float = 2.0) -> float:
        """Ratchet a stop toward price; never loosens."""
        atr = max(1e-9, float(atr))
        if side >= 0:
            return max(current_stop, price - mult * atr)
        return min(current_stop, price + mult * atr)

    # --------------------------------------------------------- governors
    def update_equity(self, equity: float) -> dict:
        """Track peak/day-start; halt on daily-loss or max-drawdown breach."""
        equity = float(equity)
        if self.peak is None:
            self.peak = equity
            self.day_start = equity
        self.peak = max(self.peak, equity)
        dd = (equity - self.peak) / (self.peak + 1e-12)
        day = (equity - (self.day_start or equity)) / (
            (self.day_start or equity) + 1e-12)
        if day <= self.daily_halt or dd <= self.max_dd:
            self.halted = True
        depth = abs(min(0.0, dd))
        soft = abs(self.max_dd) * 0.4
        if depth <= soft:
            throttle = 1.0
        elif depth >= abs(self.max_dd):
            throttle = 0.0
        else:
            x = (depth - soft) / (abs(self.max_dd) - soft + 1e-12)
            throttle = float(0.5 + 0.5 * _np.cos(_np.pi * x))
        return {"drawdown": float(dd), "day_return": float(day),
                "throttle": throttle, "halted": self.halted}

    def reset_day(self, equity: float) -> None:
        self.day_start = float(equity)
        if self.halted:
            self.halted = False

    def allow_trade(self, equity: float, current_exposure: float,
                    correlation: float = 0.0, max_corr: float = 0.75) -> dict:
        """Pre-trade gate: governors, exposure cap, correlation brake."""
        gov = self.update_equity(equity)
        if gov["halted"]:
            return {"allow": False, "reason": "governor-halted", **gov}
        if abs(current_exposure) >= self.exposure_cap:
            return {"allow": False, "reason": "exposure-cap", **gov}
        if abs(correlation) > max_corr and abs(current_exposure) > 0.3:
            return {"allow": False, "reason": "correlation-brake", **gov}
        return {"allow": True, "reason": "ok", **gov}

    def position_heat(self, positions: list) -> dict:
        """Gross/net notional across a position list."""
        gross = float(sum(abs(p.get("notional", 0.0)) for p in positions))
        net = float(sum(p.get("notional", 0.0)
                        * (1 if p.get("side", 1) > 0 else -1)
                        for p in positions))
        return {"gross": gross, "net": net, "count": len(positions)}


def position_size(equity: float, entry: float, atr: float,
                  profile: str = "default", side: int = 1) -> dict:
    """One-call position sizing under a named risk profile.

    Returns the RiskManager's ``size_position`` dict plus the ATR stop
    ladder (``stop``, ``target_1``, ``target_2``) for ``side`` (+1 long,
    -1 short) at ``entry``.
    """
    cfg = PROFILES.get((profile or "default").strip().lower(),
                       PROFILES["default"])
    rm = RiskManager(cfg)
    sizing = rm.size_position(equity, entry, atr)
    sizing["stops"] = rm.stop_levels(1 if side >= 0 else -1, entry, atr)
    sizing["profile"] = (profile or "default").strip().lower()
    return sizing


# ── sweep additions: live position management + portfolio sizing ─────────

class Position:
    """One open position with its full risk plan attached."""

    def __init__(self, symbol: str, side: int, entry: float, shares: float,
                 stop: float, atr: float, target_1: float | None = None,
                 target_2: float | None = None, time_stop_bars: int = 0,
                 breakeven_trigger: float | None = None):
        self.symbol = symbol
        self.side = 1 if side >= 0 else -1
        self.entry = float(entry)
        self.shares = float(shares)
        self.stop = float(stop)
        self.initial_stop = float(stop)
        self.atr = max(1e-9, float(atr))
        self.target_1 = target_1
        self.target_2 = target_2
        self.time_stop_bars = int(time_stop_bars)
        self.breakeven_trigger = breakeven_trigger
        self.bars_held = 0
        self.moved_to_breakeven = False
        self.partials_taken = 0
        self.highest = float(entry)   # highest favorable excursion (long)
        self.lowest = float(entry)

    @property
    def notional(self) -> float:
        return self.shares * self.entry

    @property
    def risk(self) -> float:
        """Dollars at risk if the current stop fills."""
        return self.shares * abs(self.entry - self.stop)

    def unrealized(self, price: float) -> float:
        return self.side * self.shares * (float(price) - self.entry)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side, "entry": self.entry,
            "shares": self.shares, "stop": self.stop,
            "initial_stop": self.initial_stop, "target_1": self.target_1,
            "target_2": self.target_2, "bars_held": self.bars_held,
            "risk": self.risk,
            "moved_to_breakeven": self.moved_to_breakeven,
        }


class PositionTracker:
    """Live position state machine: breakeven, trailing, time stops.

    The missing half of the old risk module — it had sizing and static
    stop ladders but nothing that *managed* a position bar by bar.
    ``update(bar)`` returns events: ``breakeven``, ``trailing``,
    ``time_stop``, ``stop_hit``, ``target_1``/``target_2``.
    """

    def __init__(self, trailing_atr_mult: float = 2.0):
        self.positions: dict[str, Position] = {}
        self.trailing_atr_mult = float(trailing_atr_mult)
        self.events: list[dict] = []

    def open(self, position: Position) -> None:
        self.positions[position.symbol] = position

    def close(self, symbol: str) -> Position | None:
        return self.positions.pop(symbol, None)

    def update(self, symbol: str, bar: dict) -> list[dict]:
        """Advance one bar: ``{"high","low","close","atr"}``. Returns events."""
        p = self.positions.get(symbol)
        if p is None:
            return []
        h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
        atr_v = max(1e-9, float(bar.get("atr", p.atr)))
        p.bars_held += 1
        p.highest = max(p.highest, h)
        p.lowest = min(p.lowest, l)
        events: list[dict] = []

        def _ev(kind: str, **kw):
            ev = {"symbol": symbol, "kind": kind, "price": c, **kw}
            events.append(ev)
            self.events.append(ev)

        if p.side > 0:
            if p.breakeven_trigger and not p.moved_to_breakeven \
                    and h >= p.breakeven_trigger:
                p.stop = max(p.stop, p.entry)
                p.moved_to_breakeven = True
                _ev("breakeven", stop=p.stop)
            new_stop = max(p.stop, p.highest - self.trailing_atr_mult * atr_v)
            if new_stop > p.stop + 1e-12:
                p.stop = new_stop
                _ev("trailing", stop=p.stop)
            if l <= p.stop:
                _ev("stop_hit", stop=p.stop)
            if p.target_1 and h >= p.target_1 and p.partials_taken == 0:
                p.partials_taken = 1
                _ev("target_1", target=p.target_1)
            if p.target_2 and h >= p.target_2 and p.partials_taken == 1:
                p.partials_taken = 2
                _ev("target_2", target=p.target_2)
        else:
            if p.breakeven_trigger and not p.moved_to_breakeven \
                    and l <= p.breakeven_trigger:
                p.stop = min(p.stop, p.entry)
                p.moved_to_breakeven = True
                _ev("breakeven", stop=p.stop)
            new_stop = min(p.stop, p.lowest + self.trailing_atr_mult * atr_v)
            if new_stop < p.stop - 1e-12:
                p.stop = new_stop
                _ev("trailing", stop=p.stop)
            if h >= p.stop:
                _ev("stop_hit", stop=p.stop)
            if p.target_1 and l <= p.target_1 and p.partials_taken == 0:
                p.partials_taken = 1
                _ev("target_1", target=p.target_1)
            if p.target_2 and l <= p.target_2 and p.partials_taken == 1:
                p.partials_taken = 2
                _ev("target_2", target=p.target_2)
        if p.time_stop_bars and p.bars_held >= p.time_stop_bars:
            _ev("time_stop", bars=p.bars_held)
        return events

    def heat(self, equity: float) -> dict:
        """Portfolio heat: gross/net exposure + total dollars at risk."""
        return portfolio_heat(list(self.positions.values()), equity)


def portfolio_heat(positions: list, equity: float) -> dict:
    """Gross/net notional, count, and summed stop-risk across positions."""
    gross = float(sum(abs(p.notional if isinstance(p, Position)
                          else p.get("notional", 0.0)) for p in positions))
    net = float(sum((p.side if isinstance(p, Position)
                     else (1 if p.get("side", 1) > 0 else -1))
                    * (p.notional if isinstance(p, Position)
                       else abs(p.get("notional", 0.0)))
                    for p in positions))
    at_risk = float(sum(p.risk if isinstance(p, Position) else 0.0
                        for p in positions))
    eq = max(1e-12, float(equity))
    return {
        "gross": gross, "net": net, "count": len(positions),
        "at_risk": at_risk,
        "gross_pct": gross / eq * 100.0,
        "at_risk_pct": at_risk / eq * 100.0,
    }


def risk_parity_weights(volatilities: dict[str, float]) -> dict[str, float]:
    """Inverse-volatility weights — each position contributes equal risk."""
    vols = {k: max(1e-9, float(v)) for k, v in volatilities.items()}
    inv = {k: 1.0 / v for k, v in vols.items()}
    tot = sum(inv.values()) or 1.0
    return {k: v / tot for k, v in inv.items()}


def correlation_adjusted_fraction(base_fraction: float,
                                  avg_correlation: float,
                                  max_corr: float = 0.7) -> float:
    """Shrink size when the new trade correlates with the book.

    At ``avg_correlation >= max_corr`` the trade adds no diversification —
    size halves; at zero correlation it passes through untouched.
    """
    c = _np.clip(abs(float(avg_correlation)), 0.0, 1.0)
    m = _np.clip(float(max_corr), 0.05, 1.0)
    shrink = 1.0 - 0.5 * _np.clip(c / m, 0.0, 1.0)
    return float(base_fraction * shrink)


def chandelier_exit(df, side: int, period: int = 22,
                    mult: float = 3.0) -> _pd.Series:
    """Chandelier exit (Chuck LeBeau): highest high − k·ATR trailing stop.

    Hangs the stop from the extreme — tighter than a fixed-ATR stop in
    trends, and it never moves against the position by construction.
    """
    from .math import atr as _atr_fn
    from .math import ensure_ohlcv as _ensure

    df = _ensure(df)
    a = _atr_fn(df, 14)
    if side >= 0:
        raw = df["high"].astype(float).rolling(
            max(2, int(period)), min_periods=1).max() - float(mult) * a
    else:
        raw = df["low"].astype(float).rolling(
            max(2, int(period)), min_periods=1).min() + float(mult) * a
    # Ratchet: longs only rise, shorts only fall.
    out = raw.copy()
    vals = raw.to_numpy()
    res = _np.empty(len(vals))
    res[0] = vals[0]
    if side >= 0:
        for i in range(1, len(vals)):
            res[i] = max(vals[i], res[i - 1])
    else:
        for i in range(1, len(vals)):
            res[i] = min(vals[i], res[i - 1])
    return _pd.Series(res, index=df.index, name="chandelier").bfill()


def optimal_f(trades: list[float]) -> dict:
    """Vince's optimal f: the fraction maximizing geometric growth.

    Returns ``f`` (fraction of equity to risk), the ``biggest_loss`` it
    was normalized against, and the geometric mean at that f.
    """
    t = _np.asarray(list(trades), dtype=float)
    if len(t) < 5:
        return {"f": 0.0, "biggest_loss": 0.0, "geo_mean": 0.0,
                "note": "need >= 5 trades"}
    biggest_loss = abs(float(t.min()))
    if biggest_loss <= 1e-12:
        return {"f": 0.0, "biggest_loss": 0.0, "geo_mean": 0.0,
                "note": "no losing trades"}
    hpr = t / biggest_loss  # holding-period returns in loss units
    best_f, best_g = 0.01, -_np.inf
    for f in _np.linspace(0.01, 1.0, 100):
        ghpr = _np.prod(1.0 + f * hpr) ** (1.0 / len(hpr))
        if ghpr > best_g:
            best_g, best_f = ghpr, f
    return {"f": float(best_f), "biggest_loss": biggest_loss,
            "geo_mean": float(best_g)}


def kelly_drawdown_shrink(kelly_f: float, current_dd: float,
                          max_dd: float) -> float:
    """Shrink Kelly as drawdown deepens — the tradable version of Kelly.

    Full Kelly at zero drawdown, linearly to zero at ``max_dd``. The old
    code had a fixed quarter-Kelly; this breathes with the equity curve.
    """
    dd = abs(min(0.0, float(current_dd)))
    cap = abs(float(max_dd)) or 0.15
    shrink = _np.clip(1.0 - dd / cap, 0.0, 1.0)
    return float(max(0.0, float(kelly_f)) * shrink)
