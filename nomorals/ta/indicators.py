"""Canonical technical indicators with exact, textbook math.

A small, hand-written set (RSI, MACD, Bollinger, Stochastic, ATR, ADX, OBV,
Donchian, VWAP, EMA/SMA via ``math``, Ichimoku, Parabolic SAR, CCI,
Williams %R, Fibonacci retracements, Keltner) — each independently
verifiable. All outputs are aligned to ``df.index`` and free of NaNs after
warmup (warmup is backfilled, never forward-leaked).
"""

from __future__ import annotations


from .math import atr as _atr
from .math import ema, ensure_ohlcv, rsi as _rsi, sma, true_range


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



__all__ = [
    "rsi",
    "macd",
    "bollinger",
    "stochastic",
    "atr",
    "adx",
    "obv",
    "donchian",
    # Re-exported from math (no duplication): the series-level primitives
    # everything here is built on.
    "ema",
    "sma",
    "vwap",
    "ichimoku",
    "psar",
    "cci",
    "williams_r",
    "fibonacci",
    "keltner",
]


def rsi(df: _pd.DataFrame, period: int = 14) -> _pd.Series:
    """Wilder RSI of close, in [0, 100]."""
    df = ensure_ohlcv(df)
    return _rsi(df["close"], period)


def macd(df: _pd.DataFrame, fast: int = 12, slow: int = 26,
         signal: int = 9) -> _pd.DataFrame:
    """MACD line, signal line and histogram (EMA-based, adjust=False)."""
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    out = _pd.DataFrame(
        {"macd": line, "signal": sig, "hist": line - sig}, index=df.index
    )
    return out.bfill().fillna(0.0)


def bollinger(df: _pd.DataFrame, period: int = 20,
              mult: float = 2.0) -> _pd.DataFrame:
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
    out = _pd.DataFrame(
        {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b,
         "width": width},
        index=df.index,
    )
    return out.fillna(0.0)


def stochastic(df: _pd.DataFrame, k: int = 14, d: int = 3) -> _pd.DataFrame:
    """Stochastic oscillator: %K in [0, 100] and its SMA %D."""
    df = ensure_ohlcv(df)
    k = max(2, int(k))
    high, low = df["high"].astype(float), df["low"].astype(float)
    close = df["close"].astype(float)
    hh = high.rolling(k, min_periods=1).max()
    ll = low.rolling(k, min_periods=1).min()
    pct_k = 100.0 * (close - ll) / ((hh - ll) + 1e-12)
    pct_d = sma(pct_k, max(2, int(d)))
    out = _pd.DataFrame({"k": pct_k.clip(0, 100), "d": pct_d.clip(0, 100)},
                       index=df.index)
    return out.bfill().fillna(50.0)


def atr(df: _pd.DataFrame, period: int = 14) -> _pd.Series:
    """Average true range (Wilder smoothing)."""
    df = ensure_ohlcv(df)
    return _atr(df, period)


def adx(df: _pd.DataFrame, period: int = 14) -> _pd.DataFrame:
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
    out = _pd.DataFrame(
        {"adx": adx_s.clip(0, 100), "plus_di": plus_di.clip(0, 100),
         "minus_di": minus_di.clip(0, 100)},
        index=df.index,
    )
    return out.bfill().fillna(0.0)


def obv(df: _pd.DataFrame) -> _pd.Series:
    """On-balance volume: cumulative signed volume."""
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    direction = _np.sign(close.diff().fillna(0.0))
    return (direction * vol).cumsum().rename("obv")


def donchian(df: _pd.DataFrame, period: int = 20) -> _pd.DataFrame:
    """Donchian channel: rolling highest high / lowest low and midline."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    high, low = df["high"].astype(float), df["low"].astype(float)
    upper = high.rolling(period, min_periods=1).max()
    lower = low.rolling(period, min_periods=1).min()
    out = _pd.DataFrame(
        {"upper": upper, "mid": (upper + lower) / 2.0, "lower": lower},
        index=df.index,
    )
    return out.bfill().ffill()


def vwap(df: _pd.DataFrame, anchor: str = "day") -> _pd.Series:
    """Volume-weighted average price, anchored per calendar day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume), reset
    at each new day when ``df`` has a DatetimeIndex (the standard session
    anchor; for 24/7 crypto this is the UTC-day VWAP). With any other index
    the anchor is the whole series. Volume must be nonzero within an
    anchor; flat-zero-volume stretches inherit the last good VWAP.
    """
    df = ensure_ohlcv(df)
    tp = (df["high"].astype(float) + df["low"].astype(float)
          + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=0.0)
    pv = (tp * vol).cumsum()
    cumvol = vol.cumsum()
    if anchor == "day" and isinstance(df.index, _pd.DatetimeIndex):
        day = df.index.floor("D")
        cumvol = vol.groupby(day).cumsum()
        pv = (tp * vol).groupby(day).cumsum()
    out = pv / (cumvol + 1e-12)
    out = out.mask(cumvol < 1e-12).ffill().bfill()
    # All-zero volume (e.g. synthetic no-volume feeds): fall back to the
    # typical price so the series stays dense and price-sensible.
    return out.fillna(tp).rename("vwap")


def ichimoku(df: _pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b_period: int = 52, displacement: int = 26
             ) -> _pd.DataFrame:
    """Ichimoku Cloud: tenkan, kijun, senkou A/B (shifted forward), chikou.

    The cloud (senkou A/B) is plotted ``displacement`` bars ahead and the
    chikou (lagging) line ``displacement`` bars behind, per the textbook
    definition. NaN displacement edges are backfilled so the frame stays
    dense; signal logic should use current-bar columns only.
    """
    df = ensure_ohlcv(df)
    high, low = df["high"].astype(float), df["low"].astype(float)
    close = df["close"].astype(float)
    tenkan_p = max(2, int(tenkan))
    kijun_p = max(2, int(kijun))
    sb_p = max(2, int(senkou_b_period))
    disp = max(0, int(displacement))
    tenkan_s = (high.rolling(tenkan_p, min_periods=1).max()
                + low.rolling(tenkan_p, min_periods=1).min()) / 2.0
    kijun_s = (high.rolling(kijun_p, min_periods=1).max()
               + low.rolling(kijun_p, min_periods=1).min()) / 2.0
    senkou_a = ((tenkan_s + kijun_s) / 2.0).shift(disp)
    senkou_b = ((high.rolling(sb_p, min_periods=1).max()
                 + low.rolling(sb_p, min_periods=1).min()) / 2.0).shift(disp)
    chikou = close.shift(-disp)
    out = _pd.DataFrame(
        {"tenkan": tenkan_s, "kijun": kijun_s, "senkou_a": senkou_a,
         "senkou_b": senkou_b, "chikou": chikou},
        index=df.index,
    )
    return out.bfill().ffill()


def psar(df: _pd.DataFrame, accel: float = 0.02,
         max_accel: float = 0.20) -> _pd.Series:
    """Wilder Parabolic SAR: iterative, bounded acceleration.

    In an uptrend the SAR trails below price (never above the prior two
    lows); a close through the SAR flips the regime. ``accel`` seeds the
    step, doubling per new extreme up to ``max_accel``.
    """
    df = ensure_ohlcv(df)
    step = float(max(1e-4, accel))
    cap = float(max(step, max_accel))
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    n = len(df)
    sar = _np.empty(n)
    # Seed long; the first bars self-correct on the first flip.
    long = True
    ep = high[0]
    af = step
    sar[0] = low[0]
    for i in range(1, n):
        raw = sar[i - 1] + af * (ep - sar[i - 1])
        if long:
            if low[i] < raw:
                # Penetrated the raw SAR: flip to short at the extreme.
                long = False
                sar[i] = ep
                ep = low[i]
                af = step
            else:
                sar[i] = min(raw, low[i - 1], low[i])
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, cap)
        else:
            if high[i] > raw:
                # Penetrated the raw SAR: flip to long at the extreme.
                long = True
                sar[i] = ep
                ep = high[i]
                af = step
            else:
                sar[i] = max(raw, high[i - 1], high[i])
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, cap)
    return _pd.Series(sar, index=df.index, name="psar")


def cci(df: _pd.DataFrame, period: int = 20) -> _pd.Series:
    """Commodity Channel Index: (TP - SMA_TP) / (0.015 * mean deviation).

    Flat markets (zero deviation) yield 0. Values beyond ±100 mark
    statistically unusual prices for the lookback.
    """
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    tp = (df["high"].astype(float) + df["low"].astype(float)
          + df["close"].astype(float)) / 3.0
    tp_sma = tp.rolling(period, min_periods=1).mean()
    md = (tp - tp_sma).abs().rolling(period, min_periods=1).mean()
    out = (tp - tp_sma) / (0.015 * (md + 1e-12))
    return out.mask(md < 1e-12, 0.0).bfill().fillna(0.0).rename(f"cci_{period}")


def williams_r(df: _pd.DataFrame, period: int = 14) -> _pd.Series:
    """Williams %R in [-100, 0]: close relative to the highest high."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    high, low = df["high"].astype(float), df["low"].astype(float)
    close = df["close"].astype(float)
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    out = -100.0 * (hh - close) / ((hh - ll) + 1e-12)
    return out.clip(-100.0, 0.0).bfill().fillna(-50.0).rename(
        f"williams_r_{period}")


_FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fibonacci(df: _pd.DataFrame, period: int = 120) -> dict:
    """Fibonacci retracement levels for the last ``period`` bars.

    Finds the swing high/low of the window and returns the classic ratios
    as ``{"ratio_label": price}`` plus ``swing_high`` / ``swing_low``.
    Level prices are direction-agnostic (from swing low up to swing high);
    traders pick the leg that matches the current trend.
    """
    df = ensure_ohlcv(df)
    period = max(2, min(int(period), len(df)))
    win = df.iloc[-period:]
    swing_high = float(win["high"].max())
    swing_low = float(win["low"].min())
    span = swing_high - swing_low
    levels = {"swing_high": swing_high, "swing_low": swing_low}
    for r in _FIB_RATIOS:
        label = f"{r * 100:.1f}%"
        levels[label] = swing_low + r * span
    return levels


def keltner(df: _pd.DataFrame, period: int = 20, atr_period: int = 10,
            mult: float = 2.0) -> _pd.DataFrame:
    """Keltner channel: EMA(``period``) ± ``mult`` * ATR(``atr_period``).

    The EMA basis makes Keltner tighter than Bollinger in calm markets and
    the ATR width makes it adaptive to volatility regime changes.
    """
    df = ensure_ohlcv(df)
    close = df["close"].astype(float)
    mid = ema(close, max(2, int(period)))
    band = float(mult) * _atr(df, max(2, int(atr_period)))
    out = _pd.DataFrame(
        {"upper": mid + band, "mid": mid, "lower": mid - band},
        index=df.index,
    )
    return out.bfill().ffill()
