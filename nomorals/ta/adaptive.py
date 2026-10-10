"""Adaptive parameter control: volatility breathing, quantile gates, rate tuning.

Ported from ``sentinel/core/adaptive.py`` (user's own Sentinel.py bot).
The key insight: fixed lookbacks and thresholds are fragile. Parameters
should breathe with volatility — wider in chop, tighter in trend.

All classes work without pandas/numpy until actually called (lazy imports).
"""

from __future__ import annotations


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


def _require_deps() -> None:
    if not (_HAS_NUMPY and _HAS_PANDAS):
        raise TAError("numpy and pandas required: pip install nomorals[ta]")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


__all__ = [
    "VolatilityBreath", "QuantileGate", "SignalRateTuner", "DrawdownThrottle",
]


class VolatilityBreath:
    """Maps realized volatility to a lookback scale in [smin, smax].

    When vol is high relative to its median, lookbacks stretch (smoother).
    When vol is low, lookbacks shrink (more responsive). This is how
    Sentinel avoids the "fixed 14-period ATR" trap.
    """

    def __init__(self, vol_window: int = 50, median_window: int = 300,
                 smin: float = 0.5, smax: float = 2.0):
        self.vol_window = vol_window
        self.median_window = median_window
        self.smin = smin
        self.smax = smax

    def scale(self, close: "_pd.Series") -> "_pd.Series":
        """Volatility ratio series: 1.0 = normal, >1 = elevated."""
        _require_deps()
        rets = close.astype(float).pct_change().fillna(0.0)
        vol = rets.rolling(self.vol_window, min_periods=10).std()
        vol = vol.bfill().ffill().fillna(1e-9)
        med = vol.rolling(self.median_window, min_periods=20).median()
        med = med.bfill().ffill().fillna(1e-9)
        return ((vol / (med + 1e-12)).clip(self.smin, self.smax)).fillna(1.0)

    def dynamic_period(self, close: "_pd.Series", base: int,
                       lo: int = 2, hi: int = 500) -> "_pd.Series":
        """Adaptive lookback: base × vol_scale, clamped to [lo, hi]."""
        _require_deps()
        s = self.scale(close)
        return ((s * base).round().clip(lo, hi)).astype(int)


class QuantileGate:
    """Self-calibrating entry/exit gates from rolling quantiles.

    Instead of "RSI > 70 = overbought", the gate learns what "extreme"
    means for THIS symbol from its own history. No hardcoded thresholds.
    """

    def __init__(self, window: int = 300, hi_q: float = 0.8, lo_q: float = 0.2):
        self.window = window
        self.hi_q = hi_q
        self.lo_q = lo_q

    def bands(self, s: "_pd.Series") -> tuple:
        """Upper/lower adaptive bands."""
        _require_deps()
        hi = s.rolling(self.window, min_periods=30).quantile(self.hi_q)
        lo = s.rolling(self.window, min_periods=30).quantile(self.lo_q)
        return hi, lo

    def position(self, s: "_pd.Series") -> "_pd.Series":
        """Position of each value inside its own adaptive band, in [-1, 1].

        +1 = at the top of its historical range, -1 = at the bottom.
        0 = dead center. Self-normalizing across symbols.
        """
        _require_deps()
        hi, lo = self.bands(s)
        mid = (hi + lo) / 2.0
        half = ((hi - lo) / 2.0).replace(0, _np.nan).bfill().ffill().fillna(1e-9)
        return (((s - mid) / (half + 1e-12)).clip(-1, 1)).fillna(0.0)


class SignalRateTuner:
    """Integral controller holding a strategy near a target signal rate.

    Over-trading → gate multiplier rises (fewer signals).
    Starving → multiplier falls (more signals).
    Bounded and drift-safe. This is how Sentinel prevents
    strategy #47 from firing 200 times a day in chop.
    """

    def __init__(self, target_rate: float = 0.05, ki: float = 0.5,
                 lo: float = 0.25, hi: float = 4.0):
        self.target = target_rate
        self.ki = ki
        self.lo = lo
        self.hi = hi
        self.mult = 1.0

    def update(self, recent_signals) -> float:
        """Feed recent signal signs (-1/0/+1); returns the gate multiplier."""
        _require_deps()
        s = _np.sign(_np.asarray(recent_signals, dtype=float))
        if len(s) < 10:
            return self.mult
        flips = float(_np.mean(_np.abs(_np.diff(s)) > 0))
        err = flips - self.target
        self.mult = _clamp(self.mult * (1.0 + self.ki * err), self.lo, self.hi)
        return self.mult

    def reset(self) -> None:
        self.mult = 1.0


class DrawdownThrottle:
    """Smooth exposure throttle from current drawdown depth.

    Cosine interpolation: full size until soft_dd, zero at hard_dd,
    smooth ramp between. No cliff edges. This is the "drawdown governor"
    from Sentinel's risk engine — the single most important piece for
    surviving losing streaks.
    """

    def __init__(self, soft_dd: float = 0.05, hard_dd: float = 0.15):
        self.soft = abs(soft_dd)
        self.hard = abs(hard_dd)

    def factor(self, equity) -> float:
        """Throttle factor in [0, 1]. Multiply position size by this."""
        _require_deps()
        eq = _np.asarray(equity, dtype=float)
        if len(eq) < 2:
            return 1.0
        peak = _np.max(eq)
        dd = float((eq[-1] - peak) / (peak + 1e-12))
        depth = abs(min(0.0, dd))
        if depth <= self.soft:
            return 1.0
        if depth >= self.hard:
            return 0.0
        span = self.hard - self.soft + 1e-12
        x = (depth - self.soft) / span
        return float(0.5 + 0.5 * _np.cos(_np.pi * x))
