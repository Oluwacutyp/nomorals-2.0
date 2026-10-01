"""Core TA math: OHLCV validation, averages, oscillators, performance stats.

Ported from ``sentinel/utils/helpers.py`` (user's own Sentinel.py bot) with
one addition: a textbook Wilder RSI (the original only computed RSI inline
inside generated code). numpy + pandas only; no NaNs leak past warmup.
"""

from __future__ import annotations

import math as _math
import random

import numpy as np
import pandas as pd

__all__ = [
    "OHLCV",
    "ensure_ohlcv",
    "ema",
    "sma",
    "rsi",
    "true_range",
    "atr",
    "rolling_zscore",
    "rolling_quantile",
    "log_returns",
    "sharpe",
    "sortino",
    "max_drawdown",
    "profit_factor",
    "clamp",
    "softmax",
    "resample_ohlcv",
    "seed_all",
]

OHLCV = ("open", "high", "low", "close", "volume")


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a frame, sort the index, coerce dtypes, drop empty rows."""
    if df is None or len(df) == 0:
        raise ValueError("empty dataframe")
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype(float)
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype(float).fillna(0.0)
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) == 0:
        raise ValueError("empty dataframe after cleaning")
    return out


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (Wilder/recursive, ``adjust=False``)."""
    return s.astype(float).ewm(span=max(2, int(span)), adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    """Simple moving average (min_periods=1 so the head is usable)."""
    return s.astype(float).rolling(max(2, int(window)), min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100].

    Recursive Wilder smoothing (``ewm(alpha=1/period, adjust=False)``), seeded
    on the first observation. A perfectly flat series yields 50 (neutral);
    pure-up yields 100, pure-down yields 0. Warmup is backfilled.
    """
    period = max(2, int(period))
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100.0 - 100.0 / (1.0 + rs)
    flat = (avg_gain < 1e-12) & (avg_loss < 1e-12)
    out = out.mask(flat, 50.0)
    return out.bfill().fillna(50.0).rename(f"rsi_{period}")


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing). Never NaN, floored at 1e-9."""
    tr = true_range(df)
    return (
        tr.ewm(alpha=1.0 / max(2, int(n)), adjust=False)
        .mean()
        .bfill()
        .fillna(1e-9)
        .rename(f"atr_{n}")
    )


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Distance from the rolling mean in rolling standard deviations."""
    m = s.rolling(window, min_periods=max(2, window // 4)).mean()
    sd = (
        s.rolling(window, min_periods=max(2, window // 4))
        .std()
        .bfill()
        .ffill()
        .fillna(1e-9)
    )
    return ((s - m) / (sd + 1e-12)).fillna(0.0)


def rolling_quantile(s: pd.Series, q: float, window: int) -> pd.Series:
    """Rolling q-quantile, forward/back filled (adaptive thresholds)."""
    return (
        s.rolling(window, min_periods=max(5, window // 5))
        .quantile(q)
        .bfill()
        .ffill()
    )


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns, first value 0."""
    c = close.astype(float)
    return np.log(c / c.shift(1)).fillna(0.0)


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio (0 when undefined)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(r) / sd * _math.sqrt(periods))


def sortino(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only; 0 when undefined)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    dn = r[r < 0]
    if len(dn) < 2 or float(np.std(dn)) <= 1e-12:
        return 0.0
    return float(np.mean(r) / float(np.std(dn)) * _math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> dict:
    """Max drawdown (negative fraction) plus peak/trough bar indices."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return {"max_dd": 0.0, "peak": 0, "trough": 0}
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (eq - peak) / peak, 0.0)
    trough = int(np.argmin(dd))
    peak_i = int(np.argmax(eq[: trough + 1])) if trough else 0
    return {"max_dd": float(dd[trough]), "peak": peak_i, "trough": trough}


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss (inf when no losses, 0 when nothing)."""
    r = np.asarray(returns, dtype=float)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def softmax(d: dict) -> dict:
    """Softmax over a {key: score} mapping (numerically safe)."""
    if not d:
        return {}
    keys = list(d.keys())
    v = np.array([float(d[k]) for k in keys], dtype=float)
    v = v - float(np.max(v))
    e = np.exp(np.clip(v, -50, 50))
    s = float(e.sum()) or 1.0
    return {k: float(e[i] / s) for i, k in enumerate(keys)}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample bars to a higher timeframe (e.g. '4h', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    out = ensure_ohlcv(df).resample(rule).agg(agg).dropna(subset=["close"])
    return out


def seed_all(seed: int = 42) -> None:
    """Seed stdlib random + numpy (deterministic synthetic data / tests)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
