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
    # ── sweep additions: finta-breadth indicators + streaming ──
    "supertrend",
    "mfi",
    "kama",
    "aroon",
    "awesome_oscillator",
    "wma",
    "hma",
    "tema",
    "dema",
    "elder_force",
    "chaikin_money_flow",
    "stoch_rsi",
    "choppiness",
    "connors_rsi",
    "coppock",
    "smi_ergodic",
    "heikin_ashi",
    "elder_ray",
    "compute_all",
    "StreamState",
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


# ── sweep additions ──────────────────────────────────────────────────────
# Breadth mined from finta / TA-Lib canon, plus talipp's incremental idea.

def supertrend(df: _pd.DataFrame, period: int = 10,
               mult: float = 3.0) -> _pd.DataFrame:
    """Supertrend (Olivier Seban): ATR bands that flip with the trend.

    ``direction`` is +1 (uptrend, trail the lower band) / -1 (downtrend).
    A flip happens only when price closes through the active band — the
    classic retail trend system, done exactly.
    """
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    a = _atr(df, period).to_numpy()
    n = len(df)
    upper = (h + l) / 2.0 + float(mult) * a
    lower = (h + l) / 2.0 - float(mult) * a
    direction = _np.ones(n)
    final_upper = upper.copy()
    final_lower = lower.copy()
    for i in range(1, n):
        if c[i - 1] > final_upper[i - 1]:
            direction[i] = 1.0
        elif c[i - 1] < final_lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
            if direction[i] > 0 and final_lower[i] < final_lower[i - 1]:
                final_lower[i] = final_lower[i - 1]
            if direction[i] < 0 and final_upper[i] > final_upper[i - 1]:
                final_upper[i] = final_upper[i - 1]
    line = _np.where(direction > 0, final_lower, final_upper)
    return _pd.DataFrame({"direction": direction, "line": line,
                          "upper": final_upper, "lower": final_lower},
                         index=df.index)


def mfi(df: _pd.DataFrame, period: int = 14) -> _pd.Series:
    """Money Flow Index: volume-weighted RSI in [0, 100]."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    tp = (df["high"].astype(float) + df["low"].astype(float)
          + df["close"].astype(float)) / 3.0
    mf = tp * df["volume"].astype(float).clip(lower=0.0)
    pos = mf.where(tp.diff() > 0, 0.0).rolling(period, min_periods=1).sum()
    neg = mf.where(tp.diff() < 0, 0.0).rolling(period, min_periods=1).sum()
    out = 100.0 - 100.0 / (1.0 + pos / (neg + 1e-12))
    flat = (pos < 1e-12) & (neg < 1e-12)
    return out.mask(flat, 50.0).bfill().fillna(50.0).rename(f"mfi_{period}")


def kama(close: _pd.Series, period: int = 10, fast: int = 2,
         slow: int = 30) -> _pd.Series:
    """Kaufman Adaptive MA: EMA whose speed follows the efficiency ratio."""
    c = close.astype(float)
    period = max(2, int(period))
    er = (c.diff(period).abs()
          / (c.diff().abs().rolling(period, min_periods=1).sum() + 1e-12)
          ).fillna(0.0)
    fast_sc = 2.0 / (max(2, fast) + 1.0)
    slow_sc = 2.0 / (max(2, slow) + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = _pd.Series(_np.nan, index=c.index, dtype=float)
    out.iloc[0] = c.iloc[0]
    sc_v = sc.to_numpy()
    c_v = c.to_numpy()
    o_v = out.to_numpy()
    for i in range(1, len(c)):
        o_v[i] = o_v[i - 1] + sc_v[i] * (c_v[i] - o_v[i - 1])
    return _pd.Series(o_v, index=c.index, name=f"kama_{period}")


def aroon(df: _pd.DataFrame, period: int = 14) -> _pd.DataFrame:
    """Aroon up/down in [0, 100] and oscillator (up − down)."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    h, l = df["high"].astype(float), df["low"].astype(float)
    up = h.rolling(period + 1, min_periods=2).apply(
        lambda v: float(_np.argmax(v)) / period * 100.0, raw=True)
    dn = l.rolling(period + 1, min_periods=2).apply(
        lambda v: float(_np.argmin(v)) / period * 100.0, raw=True)
    out = _pd.DataFrame({"aroon_up": up, "aroon_down": dn,
                         "aroon_osc": up - dn}, index=df.index)
    return out.bfill().fillna(0.0)


def awesome_oscillator(df: _pd.DataFrame, fast: int = 5,
                       slow: int = 34) -> _pd.Series:
    """Bill Williams' Awesome Oscillator: SMA5 − SMA34 of the median price."""
    df = ensure_ohlcv(df)
    mp = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
    out = sma(mp, max(2, int(fast))) - sma(mp, max(2, int(slow)))
    return out.bfill().fillna(0.0).rename("awesome_osc")


def wma(s: _pd.Series, window: int) -> _pd.Series:
    """Weighted moving average (linear weights)."""
    c = s.astype(float)
    w = max(2, int(window))
    weights = _np.arange(1, w + 1, dtype=float)
    out = c.rolling(w, min_periods=1).apply(
        lambda v: float(_np.dot(v, weights[-len(v):])
                        / weights[-len(v):].sum()), raw=True)
    return out.bfill()


def hma(s: _pd.Series, window: int) -> _pd.Series:
    """Hull MA: WMA(sqrt(n)) of (2·WMA(n/2) − WMA(n)) — minimal lag."""
    c = s.astype(float)
    w = max(2, int(window))
    inner = 2.0 * wma(c, w // 2) - wma(c, w)
    return wma(inner, int(_np.sqrt(w))).bfill()


def tema(s: _pd.Series, window: int) -> _pd.Series:
    """Triple EMA: 3·EMA − 3·EMA(EMA) + EMA(EMA(EMA)) — lag-reduced."""
    c = s.astype(float)
    w = max(2, int(window))
    e1 = ema(c, w)
    e2 = ema(e1, w)
    e3 = ema(e2, w)
    return (3.0 * e1 - 3.0 * e2 + e3).rename(f"tema_{w}")


def dema(s: _pd.Series, window: int) -> _pd.Series:
    """Double EMA: 2·EMA − EMA(EMA)."""
    c = s.astype(float)
    w = max(2, int(window))
    e1 = ema(c, w)
    return (2.0 * e1 - ema(e1, w)).rename(f"dema_{w}")


def elder_force(df: _pd.DataFrame, period: int = 13) -> _pd.Series:
    """Elder's Force Index: EMA(volume × (close − prev close))."""
    df = ensure_ohlcv(df)
    raw = df["volume"].astype(float).clip(lower=0.0) * df["close"].astype(
        float).diff().fillna(0.0)
    return ema(raw, max(2, int(period))).bfill().fillna(0.0
                                                       ).rename(
        f"force_{period}")


def chaikin_money_flow(df: _pd.DataFrame, period: int = 20) -> _pd.Series:
    """Chaikin Money Flow: Σ(MF volume) / Σ(volume) in [-1, 1]."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    h, l = df["high"].astype(float), df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float).clip(lower=0.0)
    mfm = ((c - l) - (h - c)) / ((h - l) + 1e-12)
    mfv = mfm * v
    out = (mfv.rolling(period, min_periods=1).sum()
           / (v.rolling(period, min_periods=1).sum() + 1e-12))
    return out.bfill().fillna(0.0).rename(f"cmf_{period}")


def stoch_rsi(df: _pd.DataFrame, period: int = 14, k: int = 14,
              d: int = 3) -> _pd.DataFrame:
    """Stochastic RSI: Stoch(%K of RSI) — hyper-sensitive momentum."""
    r = _rsi(df["close"] if "close" in df else df, period)
    period = max(2, int(period))
    k = max(2, int(k))
    lo = r.rolling(k, min_periods=1).min()
    hi = r.rolling(k, min_periods=1).max()
    pct_k = 100.0 * (r - lo) / ((hi - lo) + 1e-12)
    pct_d = sma(pct_k, max(2, int(d)))
    out = _pd.DataFrame({"stochrsi_k": pct_k.clip(0, 100),
                         "stochrsi_d": pct_d.clip(0, 100)}, index=df.index)
    return out.bfill().fillna(50.0)


def choppiness(df: _pd.DataFrame, period: int = 14) -> _pd.Series:
    """Choppiness Index in [0, 100]: high = chop, low = trending."""
    df = ensure_ohlcv(df)
    period = max(2, int(period))
    tr = true_range(df)
    atr_sum = tr.rolling(period, min_periods=1).sum()
    h, l = df["high"].astype(float), df["low"].astype(float)
    hl_range = h.rolling(period, min_periods=1).max() - l.rolling(
        period, min_periods=1).min()
    out = 100.0 * _np.log10(atr_sum / (hl_range + 1e-12)) / _np.log10(period)
    return out.clip(0, 100).bfill().fillna(50.0).rename(f"chop_{period}")


def connors_rsi(df: _pd.DataFrame, rsi_p: int = 3, streak_p: int = 2,
                roc_p: int = 100) -> _pd.Series:
    """Connors RSI: mean(RSI(3), streak-RSI(2), ROC-percentile) in [0, 100].

    The most statistically documented short-term mean-reversion gauge
    (Larry Connors). <10 = washed out, >90 = overbought extreme.
    """
    df = ensure_ohlcv(df)
    c = df["close"].astype(float)
    r1 = _rsi(c, max(2, int(rsi_p)))
    chg = _np.sign(c.diff().fillna(0.0)).to_numpy()
    streak = _np.zeros(len(c))
    s = 0.0
    for i in range(len(c)):
        s = s + chg[i] if _np.sign(s) == chg[i] or s == 0 else chg[i]
        streak[i] = s
    streak_s = _pd.Series(streak, index=c.index)
    r2 = _rsi(streak_s, max(2, int(streak_p)))
    roc = c.pct_change(1).fillna(0.0)
    pct = roc.rolling(max(10, int(roc_p)), min_periods=5).rank(pct=True) * 100.0
    out = (r1 + r2 + pct.fillna(50.0)) / 3.0
    return out.bfill().fillna(50.0).rename("connors_rsi")


def coppock(df: _pd.DataFrame, roc1: int = 14, roc2: int = 11,
            wma_p: int = 10) -> _pd.Series:
    """Coppock Curve: WMA(ROC14 + ROC11) — long-term momentum turns."""
    df = ensure_ohlcv(df)
    c = df["close"].astype(float)
    roc = c.pct_change(max(1, int(roc1))).fillna(0.0) * 100.0 \
        + c.pct_change(max(1, int(roc2))).fillna(0.0) * 100.0
    return wma(roc, max(2, int(wma_p))).bfill().fillna(0.0
                                                      ).rename("coppock")


def smi_ergodic(df: _pd.DataFrame, fast: int = 5, slow: int = 20,
                signal: int = 5) -> _pd.DataFrame:
    """Blau's SMI Ergodic: double-smoothed close−median momentum + signal."""
    df = ensure_ohlcv(df)
    c = df["close"].astype(float)
    med = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
    raw = c - med
    num = ema(ema(raw, max(2, int(fast))), max(2, int(slow)))
    den = ema(ema((df["high"].astype(float) - df["low"].astype(float)).abs(),
                  max(2, int(fast))), max(2, int(slow)))
    smi = 100.0 * num / (den + 1e-12)
    sig = ema(smi, max(2, int(signal)))
    out = _pd.DataFrame({"smi": smi, "smi_signal": sig,
                         "smi_hist": smi - sig}, index=df.index)
    return out.bfill().fillna(0.0)


def heikin_ashi(df: _pd.DataFrame) -> _pd.DataFrame:
    """Heikin-Ashi candles: trend-smoothed OHLC (recursive, exact)."""
    df = ensure_ohlcv(df)
    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    n = len(df)
    ha_c = (o + h + l + c) / 4.0
    ha_o = _np.empty(n)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = _np.maximum(_np.maximum(h, ha_o), ha_c)
    ha_l = _np.minimum(_np.minimum(l, ha_o), ha_c)
    return _pd.DataFrame({"open": ha_o, "high": ha_h, "low": ha_l,
                          "close": ha_c, "volume": df["volume"].astype(
                              float).to_numpy()},
                         index=df.index)


def elder_ray(df: _pd.DataFrame, period: int = 13) -> _pd.DataFrame:
    """Elder-Ray: bull power = high − EMA, bear power = low − EMA."""
    df = ensure_ohlcv(df)
    e = ema(df["close"].astype(float), max(2, int(period)))
    out = _pd.DataFrame({
        "bull_power": df["high"].astype(float) - e,
        "bear_power": df["low"].astype(float) - e,
    }, index=df.index)
    return out.bfill().fillna(0.0)


def compute_all(df: _pd.DataFrame) -> dict[str, _pd.Series | _pd.DataFrame]:
    """Every indicator in one call — the ML feature-matrix builder.

    Returns ``{name: Series/DataFrame}``. Scalar oscillators come back as
    Series, multi-output indicators as DataFrames.
    """
    df = ensure_ohlcv(df)
    out: dict = {}
    out["rsi"] = rsi(df)
    out["macd"] = macd(df)
    out["bollinger"] = bollinger(df)
    out["stochastic"] = stochastic(df)
    out["atr"] = atr(df)
    out["adx"] = adx(df)
    out["obv"] = obv(df)
    out["donchian"] = donchian(df)
    out["vwap"] = vwap(df)
    out["ichimoku"] = ichimoku(df)
    out["psar"] = psar(df)
    out["cci"] = cci(df)
    out["williams_r"] = williams_r(df)
    out["keltner"] = keltner(df)
    out["supertrend"] = supertrend(df)
    out["mfi"] = mfi(df)
    out["kama"] = kama(df["close"])
    out["aroon"] = aroon(df)
    out["awesome_osc"] = awesome_oscillator(df)
    out["elder_force"] = elder_force(df)
    out["cmf"] = chaikin_money_flow(df)
    out["stoch_rsi"] = stoch_rsi(df)
    out["choppiness"] = choppiness(df)
    out["connors_rsi"] = connors_rsi(df)
    out["coppock"] = coppock(df)
    out["smi"] = smi_ergodic(df)
    out["elder_ray"] = elder_ray(df)
    return out


class StreamState:
    """Exact O(1) incremental indicators for the live bot (talipp's idea).

    Recomputing a full 2000-bar frame every tick is O(n) waste on a phone.
    EMA / Wilder-RSI / ATR / MACD all have *exact* recursive forms, so the
    streaming values match the batch values to machine precision.

    Usage:
        st = StreamState.from_frame(df)   # seed from history
        st.update(bar)                    # new closed bar -> dict of values
        st.update_last(bar)               # replace the forming bar
    """

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 rsi_period: int = 14, atr_period: int = 14,
                 macd_signal: int = 9):
        self.ema_fast_n = max(2, int(ema_fast))
        self.ema_slow_n = max(2, int(ema_slow))
        self.rsi_n = max(2, int(rsi_period))
        self.atr_n = max(2, int(atr_period))
        self.macd_sig_n = max(2, int(macd_signal))
        self._reset()

    def _reset(self):
        self.n = 0
        self.prev_close = None
        self.ema_fast = self.ema_slow = None
        self.macd_sig = None
        self.rsi_gain = self.rsi_loss = None
        self.atr = None
        self.last = {}
        # Pre-step state of the most recent bar — lets update_last() pop
        # the last bar and re-step exactly (still O(1)).
        self._prev_state = None

    def _state_tuple(self):
        return (self.n, self.prev_close, self.ema_fast, self.ema_slow,
                self.macd_sig, self.rsi_gain, self.rsi_loss, self.atr,
                dict(self.last))

    def _restore(self, t):
        (self.n, self.prev_close, self.ema_fast, self.ema_slow,
         self.macd_sig, self.rsi_gain, self.rsi_loss, self.atr,
         self.last) = t

    @classmethod
    def from_frame(cls, df: _pd.DataFrame, **kw) -> "StreamState":
        """Seed the recursive state from a history frame (exact match)."""
        df = ensure_ohlcv(df)
        st = cls(**kw)
        c = df["close"].astype(float)
        st.ema_fast = float(ema(c, st.ema_fast_n).iloc[-1])
        st.ema_slow = float(ema(c, st.ema_slow_n).iloc[-1])
        line = ema(c, st.ema_fast_n) - ema(c, st.ema_slow_n)
        st.macd_sig = float(ema(line, st.macd_sig_n).iloc[-1])
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        st.rsi_gain = float(gain.ewm(alpha=1.0 / st.rsi_n,
                                     adjust=False).mean().iloc[-1])
        st.rsi_loss = float(loss.ewm(alpha=1.0 / st.rsi_n,
                                     adjust=False).mean().iloc[-1])
        st.atr = float(_atr(df, st.atr_n).iloc[-1])
        st.prev_close = float(c.iloc[-1])
        st.n = len(df)
        st.last = st._snapshot(float(c.iloc[-1]))
        st._prev_state = None  # no pop available until first update()
        return st

    def _snapshot(self, close: float) -> dict:
        macd_line = (self.ema_fast or 0.0) - (self.ema_slow or 0.0)
        rs = (self.rsi_gain or 0.0) / ((self.rsi_loss or 0.0) + 1e-12)
        if (self.rsi_gain or 0.0) < 1e-12 and (self.rsi_loss or 0.0) < 1e-12:
            rsi_v = 50.0
        else:
            rsi_v = 100.0 - 100.0 / (1.0 + rs)
        return {
            "close": close,
            f"ema_{self.ema_fast_n}": self.ema_fast,
            f"ema_{self.ema_slow_n}": self.ema_slow,
            "macd": macd_line,
            "macd_signal": self.macd_sig,
            "macd_hist": macd_line - (self.macd_sig or 0.0),
            f"rsi_{self.rsi_n}": rsi_v,
            f"atr_{self.atr_n}": self.atr,
        }

    def _step(self, o: float, h: float, l: float, c: float):
        self._prev_state = self._state_tuple()
        kf = 2.0 / (self.ema_fast_n + 1.0)
        ks = 2.0 / (self.ema_slow_n + 1.0)
        ksig = 2.0 / (self.macd_sig_n + 1.0)
        kr = 1.0 / self.rsi_n
        ka = 1.0 / self.atr_n
        if self.n == 0:
            self.ema_fast = self.ema_slow = c
            self.macd_sig = 0.0
            self.rsi_gain = self.rsi_loss = 0.0
            self.atr = h - l
        else:
            self.ema_fast = c * kf + self.ema_fast * (1 - kf)
            self.ema_slow = c * ks + self.ema_slow * (1 - ks)
            macd_line = self.ema_fast - self.ema_slow
            self.macd_sig = macd_line * ksig + self.macd_sig * (1 - ksig)
            d = c - self.prev_close
            self.rsi_gain = max(d, 0.0) * kr + self.rsi_gain * (1 - kr)
            self.rsi_loss = max(-d, 0.0) * kr + self.rsi_loss * (1 - kr)
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            self.atr = tr * ka + self.atr * (1 - ka)
        self.prev_close = c
        self.n += 1
        self.last = self._snapshot(c)
        return dict(self.last)

    def update(self, bar: dict) -> dict:
        """Append one closed bar: ``{"open","high","low","close"}``."""
        return self._step(float(bar["open"]), float(bar["high"]),
                          float(bar["low"]), float(bar["close"]))

    def update_last(self, bar: dict) -> dict:
        """Replace the forming bar and recompute — exact, O(1).

        Pops the most recent bar's pre-step state, then steps forward with
        ``bar``. Only the latest bar can be replaced (call once per tick).
        """
        if self._prev_state is not None:
            self._restore(self._prev_state)
        return self._step(float(bar["open"]), float(bar["high"]),
                          float(bar["low"]), float(bar["close"]))

    @property
    def value(self) -> dict:
        """Latest indicator values."""
        return dict(self.last)

