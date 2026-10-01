"""Canonical technical indicators with exact, textbook math.

A small, hand-written set (RSI, MACD, Bollinger, Stochastic, ATR, ADX, OBV,
Donchian) — each independently verifiable — instead of Sentinel's
auto-generated indicator zoo. All outputs are aligned to ``df.index`` and
free of NaNs after warmup (warmup is backfilled, never forward-leaked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .math import atr as _atr
from .math import ema, ensure_ohlcv, rsi as _rsi, sma, true_range

__all__ = [
    "rsi",
    "macd",
    "bollinger",
    "stochastic",
    "atr",
    "adx",
    "obv",
    "donchian",
]


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI of close, in [0, 100]."""
    df = ensure_ohlcv(df)
    return _rsi(df["close"], period)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram (EMA-based, adjust=False)."""
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    out = pd.DataFrame(
        {"macd": line, "signal": sig, "hist": line - sig}, index=df.index
    )
    return out.bfill().fillna(0.0)


def bollinger(df: pd.DataFrame, period: int = 20,
              mult: float = 2.0) -> pd.DataFrame:
    """Bollinger bands (population std, ddof=0) plus %B and bandwidth."""
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    period = max(2, int(period))
    mid = sma(close, period)
    sd = close.rolling(period, min_periods=1).std(ddof=0).bfill().fillna(0.0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    width = upper - lower
    pct_b = ((close - lower) / (width + 1e-12)).clip(0.0, 1.0)
    out = pd.DataFrame(
        {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b,
         "width": width},
        index=df.index,
    )
    return out.fillna(0.0)


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Stochastic oscillator: %K in [0, 100] and its SMA %D."""
    df = ensure_ohlcv(df)
    k = max(2, int(k))
    high, low = df["high"].astype(float), df["low"].astype(float)
    close = df["close"].astype(float)
    hh = high.rolling(k, min_periods=1).max()
    ll = low.rolling(k, min_periods=1).min()
    pct_k = 100.0 * (close - ll) / ((hh - ll) + 1e-12)
    pct_d = sma(pct_k, max(2, int(d)))
    out = pd.DataFrame({"k": pct_k.clip(0, 100), "d": pct_d.clip(0, 100)},
                       index=df.index)
    return out.bfill().fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing)."""
    df = ensure_ohlcv(df)
    return _atr(df, period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI. ADX in [0, 100]; DI in [0, 100]."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    high, low = df["high"].astype(float), df["low"].astype(float)
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(df)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (
        atr_s + 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (
        atr_s + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (
        (plus_di + minus_di) + 1e-12)
    adx_s = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    flat = atr_s < 1e-9
    adx_s = adx_s.mask(flat, 0.0)
    out = pd.DataFrame(
        {"adx": adx_s.clip(0, 100), "plus_di": plus_di.clip(0, 100),
         "minus_di": minus_di.clip(0, 100)},
        index=df.index,
    )
    return out.bfill().fillna(0.0)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume: cumulative signed volume."""
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * vol).cumsum().rename("obv")


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel: rolling highest high / lowest low and midline."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    high, low = df["high"].astype(float), df["low"].astype(float)
    upper = high.rolling(period, min_periods=1).max()
    lower = low.rolling(period, min_periods=1).min()
    out = pd.DataFrame(
        {"upper": upper, "mid": (upper + lower) / 2.0, "lower": lower},
        index=df.index,
    )
    return out.bfill().ffill()
